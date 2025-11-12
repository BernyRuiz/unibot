#!/usr/bin/env node
/**
 * scripts/ingest.ts (V4 - Embeddings locales optimizados)
 * Lee PDF/TXT/MD, fragmenta, crea embeddings locales (sin API) y guarda en Supabase.
 * Correr SIEMPRE desde: /frontend
 *
 * Ejemplo:
 *   npx tsx scripts/ingest.ts --file "../docs/reglamento.txt" --name "Reglamento (local)"
 */

import path from "node:path";
import fs from "node:fs/promises";
import pdf from "pdf-parse";
import { createClient } from "@supabase/supabase-js";
import dotenv from "dotenv";
import { pipeline } from "@xenova/transformers";

// 1️⃣ Cargar variables de entorno (.env.local)
dotenv.config({ path: path.resolve(process.cwd(), ".env.local") });
dotenv.config();

// 2️⃣ Variables necesarias
const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL;
const SUPABASE_ANON = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

if (!SUPABASE_URL || !SUPABASE_ANON) {
  console.error("❌ Faltan variables de entorno. Revisa .env.local:");
  console.error("   NEXT_PUBLIC_SUPABASE_URL, NEXT_PUBLIC_SUPABASE_ANON_KEY");
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON);

// 3️⃣ Configuración
const DEFAULT_CHUNK_SIZE = Number(process.env.CHUNK_SIZE || 800);
const DEFAULT_CHUNK_OVERLAP = Number(process.env.CHUNK_OVERLAP || 120);
const BATCH_SIZE = 100; // número de fragmentos procesados por lote

// 4️⃣ Leer argumentos
type Args = { file: string; name: string; url?: string; size: number; overlap: number };
function parseArgs(argv: string[]): Args {
  const get = (flag: string) => {
    const i = argv.indexOf(flag);
    return i >= 0 ? argv[i + 1] : undefined;
  };
  const file = get("--file");
  const name = get("--name");
  const url = get("--url");
  const size = Number(get("--size") ?? DEFAULT_CHUNK_SIZE);
  const overlap = Number(get("--overlap") ?? DEFAULT_CHUNK_OVERLAP);
  if (!file || !name) {
    console.error('Uso: npx tsx scripts/ingest.ts --file "<ruta>" --name "Nombre" [--url URL] [--size N] [--overlap M]');
    process.exit(1);
  }
  return { file, name, url, size, overlap };
}

// 5️⃣ Funciones auxiliares
async function readText(filePath: string): Promise<string> {
  const ext = path.extname(filePath).toLowerCase();
  const buf = await fs.readFile(filePath);
  if (ext === ".pdf") {
    try {
      const data = await pdf(buf);
      return (data.text || "").toString();
    } catch (e) {
      throw new Error("No se pudo leer el PDF. Prueba exportarlo a .txt o verifica que no esté protegido.");
    }
  }
  if (ext === ".txt" || ext === ".md") return buf.toString("utf8");
  throw new Error(`Formato no soportado: ${ext}. Usa .pdf, .txt o .md`);
}

function normalize(text: string) {
  return text
    .replace(/\r/g, "\n")
    .replace(/\t/g, " ")
    .replace(/[ \u00A0]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function slidingWindows(s: string, size: number, overlap: number): string[] {
  const out: string[] = [];
  const step = Math.max(1, size - overlap);
  for (let start = 0; start < s.length; start += step) {
    const end = Math.min(s.length, start + size);
    out.push(s.slice(start, end));
    if (end === s.length) break;
  }
  return out;
}

function chunkText(text: string, size: number, overlap: number) {
  const paras = text.split(/\n\s*\n/);
  const chunks: string[] = [];
  let buf = "";
  const flush = () => {
    const t = buf.trim();
    if (t) chunks.push(t);
    buf = "";
  };
  for (const p of paras) {
    if ((buf + "\n\n" + p).length <= size) {
      buf = buf ? buf + "\n\n" + p : p;
    } else {
      chunks.push(...slidingWindows(buf, size, overlap));
      buf = p;
    }
  }
  if (buf) {
    if (buf.length <= size) flush();
    else chunks.push(...slidingWindows(buf, size, overlap));
  }
  return chunks.map(c => c.trim()).filter(c => c.length >= Math.min(80, Math.floor(size * 0.15)));
}

// 6️⃣ Embeddings locales (Hugging Face)
let embedder: any = null;
async function embedBatch(texts: string[]): Promise<number[][]> {
  if (!embedder) {
    console.log("🧠 Cargando modelo local de embeddings (all-MiniLM-L6-v2)...");
    embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
  }

  const results: number[][] = [];
  for (const text of texts) {
    const output = await embedder(text, { pooling: "mean", normalize: true });
    results.push(Array.from(output.data));
  }
  return results;
}

async function sleep(ms: number) {
  return new Promise(r => setTimeout(r, ms));
}

// 7️⃣ Main
async function main() {
  const { file, name, url, size, overlap } = parseArgs(process.argv.slice(2));
  const absPath = path.isAbsolute(file) ? file : path.resolve(process.cwd(), file);

  console.log("📄 Archivo:", absPath);
  const raw = await readText(absPath);
  const text = normalize(raw);

  console.log("✂️  Fragmentando…");
  const pieces = chunkText(text, size, overlap);

  if (!pieces.length) {
    console.error("No se generaron fragmentos. Ajusta --size/--overlap o revisa el archivo.");
    process.exit(1);
  }

  console.log(`✅ ${pieces.length} fragmentos (size=${size}, overlap=${overlap})`);

  console.log("🗂️  Insertando en documents…");
  const { data: doc, error: docErr } = await supabase
    .from("documents")
    .insert({ name, source_url: url ?? null, uploaded_by: "ingest-script-local" })
    .select()
    .single();
  if (docErr || !doc) throw new Error("Insert documents falló: " + (docErr?.message || "sin detalle"));

  const documentId = doc.id as string;
  console.log("🧠 Embeddings → chunks…");

  for (let start = 0; start < pieces.length; start += BATCH_SIZE) {
    const batch = pieces.slice(start, start + BATCH_SIZE);
    const embeddings = await embedBatch(batch);

    const buffer = batch.map((content, i) => ({
      document_id: documentId,
      content,
      embedding: embeddings[i],
      metadata: { file: path.basename(absPath), index: start + i },
    }));

    const { error: insErr } = await supabase.from("chunks_v384").insert(buffer);
    if (insErr) throw new Error("Insert chunks falló: " + insErr.message);

    console.log(`   → guardados ${Math.min(start + BATCH_SIZE, pieces.length)}/${pieces.length}`);
    await sleep(150);
  }

  console.log("\n🎉 Ingesta completa.");
  console.log("document_id:", documentId);
}

main().catch((e) => {
  console.error("❌ Error:", e?.message || e);
  process.exit(1);
});
