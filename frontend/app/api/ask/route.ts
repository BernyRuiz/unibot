import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";
import { embedWithGemini, askGemini } from "@/lib/ai";

const TOP_K = Number(process.env.RAG_TOP_K || 5);
const CONF_THRESHOLD = Number(process.env.CONFIDENCE_THRESHOLD || 0.7);

export async function POST(req: NextRequest) {
  try {
    const { question } = await req.json();
    if (!question || typeof question !== "string") {
      return NextResponse.json({ error: "Pregunta inválida" }, { status: 400 });
    }

    // 1) Embedding de la pregunta (384D)
    const qEmbedding = await embedWithGemini(question);

    // 2) Búsqueda vectorial contra la función de 384D
    const { data: matches, error: rpcErr } = await supabase.rpc("match_chunks_v384", {
      query_embedding: qEmbedding,
      match_count: TOP_K,
    });

    if (rpcErr) {
      console.error("RPC error:", rpcErr);
      return NextResponse.json({ error: "Falló la búsqueda vectorial" }, { status: 500 });
    }

    // Si no hay contexto, responde útil sin Gemini
    if (!matches || matches.length === 0) {
      const fallback = "No tengo información suficiente en la base de conocimiento para responder esa pregunta. " +
        "Intenta con otra redacción o carga más documentos.";
      // Guarda la query aunque no haya respuesta generada
      await supabase.from("queries").insert({
        question,
        answer: fallback,
        confidence: 0,
        citations: [],
      });
      return NextResponse.json({ answer: fallback, citations: [], confidence: 0 });
    }

    // 3) Mapear documentos para citas
    const docIds: string[] = Array.from(new Set(matches.map((m: any) => m.document_id)));
    let docsById: Record<string, { name: string; source_url: string | null }> = {};
    if (docIds.length > 0) {
      const { data: docs, error: docErr } = await supabase
        .from("documents")
        .select("id, name, source_url")
        .in("id", docIds);
      if (!docErr && docs) {
        for (const d of docs) docsById[d.id] = { name: d.name, source_url: d.source_url };
      }
    }

    // 4) Contexto segmentado con encabezado por documento
    const contextBlocks: string[] = matches.map((m: any, idx: number) => {
      const doc = docsById[m.document_id] || { name: "desconocido", source_url: null };
      const header = `# [${idx + 1}] ${doc.name}`;
      return `${header}\n${m.content}`;
    });

    // Cuidar tamaño del contexto
    let context = "";
    for (const block of contextBlocks) {
      if ((context + "\n\n---\n\n" + block).length > 9000) break;
      context = context ? context + "\n\n---\n\n" + block : block;
    }

    // 5) Pedir respuesta (con fallback interno si Gemini falla)
    const answer = await askGemini(question, context);

    // 6) Confianza (heurística por similitud del top-1)
    const topSim = (matches?.[0]?.similarity as number) ?? 0;
    const confidence = Math.max(0, Math.min(1, topSim));

    // 7) Citas
    const citations = matches.map((m: any) => {
      const doc = docsById[m.document_id] || { name: "desconocido", source_url: null };
      return {
        docName: doc.name,
        sourceUrl: doc.source_url,
        snippet: m.content.slice(0, 240) + (m.content.length > 240 ? "..." : ""),
        similarity: Number(m.similarity?.toFixed(3) ?? 0),
      };
    });

    // 8) Persistir consulta y ticket opcional
    const { data: saved } = await supabase
      .from("queries")
      .insert({ question, answer, confidence, citations })
      .select()
      .single();

    if (confidence < CONF_THRESHOLD && saved?.id) {
      await supabase.from("tickets").insert({ query_id: saved.id, status: "open" });
    }

    return NextResponse.json({ answer, citations, confidence });
  } catch (e: any) {
    console.error(e);
    return NextResponse.json({ error: "Error en /api/ask" }, { status: 500 });
  }
}
