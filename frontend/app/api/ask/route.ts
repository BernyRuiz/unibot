import { NextRequest, NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";
import { embedWithGemini, askGemini } from "@/lib/ai";

const TOP_K = Number(process.env.RAG_TOP_K || 5);
const CONF_THRESHOLD = Number(process.env.CONFIDENCE_THRESHOLD || 0.55);

export async function POST(req: NextRequest) {
  try {
    const { question } = await req.json();

    if (!question || typeof question !== "string") {
      return NextResponse.json({ error: "Pregunta inválida" }, { status: 400 });
    }

    // 1) Embedding de la pregunta
    const qEmbedding = await embedWithGemini(question);

    // 2) Búsqueda vectorial (RPC en Supabase)
    const { data: matches, error: rpcErr } = await supabase.rpc("match_chunks_v384", {
      query_embedding: qEmbedding,
      match_count: TOP_K,
    });

    if (rpcErr) {
      console.error("RPC error:", rpcErr);
      return NextResponse.json({ error: "Falló la búsqueda vectorial" }, { status: 500 });
    }

    // 3) Obtener info de documentos para citas
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

    // 4) Armar contexto con encabezados por documento
    //    (mejora la trazabilidad y calidad de la respuesta)
    const contextBlocks: string[] = matches.map((m: any, idx: number) => {
      const doc = docsById[m.document_id] || { name: "desconocido", source_url: null };
      const header = `# [${idx + 1}] ${doc.name}`;
      return `${header}\n${m.content}`;
    });
    const context = contextBlocks.join("\n\n---\n\n");

    // 5) Preguntar a Gemini con el contexto
    const answer = await askGemini(question, context);

    // 6) Calcular confianza (heurística simple con similitud top-1)
    const topSim = (matches?.[0]?.similarity as number) ?? 0;
    const confidence = Math.max(0, Math.min(1, topSim));

    // 7) Armar citas (nombre doc + URL si existe)
    const citations = matches.map((m: any) => {
      const doc = docsById[m.document_id] || { name: "desconocido", source_url: null };
      return {
        docName: doc.name,
        sourceUrl: doc.source_url,
        snippet: m.content.slice(0, 240) + (m.content.length > 240 ? "..." : ""),
        similarity: Number(m.similarity?.toFixed(3) ?? 0),
      };
    });

    // 8) Persistir en queries
    const { data: saved } = await supabase
      .from("queries")
      .insert({
        question,
        answer,
        confidence,
        citations,
      })
      .select()
      .single();

    // 9) Crear ticket HITL si la confianza es baja
    if (confidence < CONF_THRESHOLD && saved?.id) {
      await supabase.from("tickets").insert({ query_id: saved.id, status: "open" });
    }

    return NextResponse.json({ answer, citations, confidence });
  } catch (e: any) {
    console.error(e);
    return NextResponse.json({ error: "Error en /api/ask" }, { status: 500 });
  }
}
