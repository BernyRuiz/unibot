#!/usr/bin/env node
/**
 * scripts/ingest.ts (V2)
 * Lee PDF/TXT/MD, fragmenta, crea embeddings (Gemini) y guarda en Supabase.
 * Correr SIEMPRE desde: /frontend
 *
 * Ejemplos:
 *   npx tsx scripts/ingest.ts --file ".\docs\faq.txt" --name "FAQ"
 *   npx tsx scripts/ingest.ts --file ".\docs\reglamento.pdf" --name "Reglamento" --url "https://www.anahuac.mx/queretaro/descargables/Compendio_Reglamentario.pdf"
 */

import path from "node:path";
import fs from "node:fs/promises";
import pdf from "pdf-parse";
import { createClient } from "@supabase/supabase-js";
import dotenv from "dotenv";

// 1) Cargar .env.local explícito 
dotenv.config({ path: path.resolve(process.cwd(), ".env.local") });
dotenv.config(); 

// 2) Variables
const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL;
const SUPABASE_ANON = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
const GEMINI_KEY = process.env.GEMINI_API_KEY;

if (!SUPABASE_URL || !SUPABASE_ANON || !GEMINI_KEY) {
  console.error("❌ Faltan variables de entorno. Revisa .env.local:");
  console.error("   NEXT_PUBLIC_SUPABASE_URL, NEXT_PUBLIC_SUPABASE_ANON_KEY, GEMINI_API_KEY");
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON);

// 3) Config
const DEFAULT_CHUNK_SIZE = Number(process.env.CHUNK_SIZE || 800);
const DEFAULT_CHUNK_OVERLAP = Number(process.env.CHUNK_OVERLAP || 120);
const EMBEDDING_URL =
  "https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedText?key=" + GEMINI_KEY;

// 4) Args
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

// 5) IO
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

async function embed(text: string): Promise<number[]> {
  const res = await fetch(
    "https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key=" + GEMINI_KEY,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "embedding-001",
        content: {
          parts: [{ text }],
        },
      }),
    }
  );

  if (!res.ok) throw new Error(`Embeddings error ${res.status}: ${await res.text()}`);
  const data = await res.json();
  const values =
    data?.embedding?.values ??
    data?.data?.[0]?.embedding?.values ??
    data?.embedding ??
    [];
  if (!values || !Array.isArray(values)) throw new Error("Embedding vacío");
  return values;
}


async function sleep(ms: number) {
  return new Promise(r => setTimeout(r, ms));
}

// 6) Main
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
    .insert({ name, source_url: url ?? null, uploaded_by: "ingest-script" })
    .select()
    .single();
  if (docErr || !doc) throw new Error("Insert documents falló: " + (docErr?.message || "sin detalle"));

  const documentId = doc.id as string;

  console.log("🧠 Embeddings → chunks…");
  const BATCH = 40;
  const buffer: any[] = [];
  for (let i = 0; i < pieces.length; i++) {
    const content = pieces[i];
    if (i > 0 && i % 12 === 0) await sleep(150);

    const embedding = await embed(content); // 768-d para text-embedding-004
    buffer.push({
      document_id: documentId,
      content,
      embedding,
      metadata: { file: path.basename(absPath), index: i },
    });

    if (buffer.length >= BATCH || i === pieces.length - 1) {
      const { error: insErr } = await supabase.from("chunks").insert(buffer);
      if (insErr) throw new Error("Insert chunks falló: " + insErr.message);
      process.stdout.write(`   → guardados ${i + 1}/${pieces.length}\r`);
      buffer.length = 0;
    }
  }

  console.log("\n🎉 Ingesta completa.");
  console.log("document_id:", documentId);
}

main().catch((e) => {
  console.error("❌ Error:", e?.message || e);
  process.exit(1);
});
