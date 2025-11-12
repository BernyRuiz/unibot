// frontend/lib/ai.ts
import { pipeline } from "@xenova/transformers";

// =============== Embeddings locales (384D) ===============
let embedder: any = null;

export async function embedWithGemini(text: string): Promise<number[]> {
  if (!embedder) {
    console.log("🧠 Cargando modelo local de embeddings (all-MiniLM-L6-v2)...");
    embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
  }
  const output = await embedder(text, { pooling: "mean", normalize: true });
  return Array.from(output.data); // 384 dims
}

// =============== Generación (Gemini con fallback) ===============
function buildLocalAnswer(question: string, context: string): string {
  // Toma los primeros bloques visibles del contexto para un resumen simple
  const parts = context.split("\n\n---\n\n").slice(0, 3);
  if (parts.length === 0) return 'No tengo información suficiente en la base de conocimiento.';
  const bullets = parts.map((p, i) => {
    const lines = p.split("\n");
    const title = lines[0]?.replace(/^#\s*\[\d+\]\s*/i, "").trim() || `Fuente ${i + 1}`;
    const snippet = lines.slice(1).join(" ").slice(0, 300);
    return `• ${title}: ${snippet}${snippet.length >= 300 ? "..." : ""}`;
  });
  return [
    "Con base en lo que encontré en los documentos:",
    "",
    ...bullets,
    "",
    "Si necesitas más detalle, dime qué sección puntual quieres que explore."
  ].join("\n");
}

export async function askGemini(question: string, context: string): Promise<string> {
  const GEMINI_KEY = process.env.GEMINI_API_KEY;
  // Limitar contexto para evitar rechazos por tamaño
  const SAFE_CONTEXT = context.slice(0, 8000);

  if (!GEMINI_KEY) {
    console.warn("⚠️ Sin GEMINI_API_KEY: usando respuesta local.");
    return buildLocalAnswer(question, SAFE_CONTEXT);
  }

  try {
    const res = await fetch(
      // API v1 estable
      "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key=" + GEMINI_KEY,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          contents: [
            {
              role: "user",
              parts: [
                {
                  text: `Eres un asistente para alumnos universitarios.
Responde SOLO con información del contexto. Si no hay suficiente evidencia, di: "No tengo información suficiente".
Cierra con referencias tipo [doc:nombre].

Pregunta: ${question}

Contexto:
${SAFE_CONTEXT}`
                }
              ]
            }
          ],
          generationConfig: { temperature: 0.85, maxOutputTokens: 512 }
        })
      }
    );

    if (!res.ok) {
      const errorText = await res.text();
      console.warn("Gemini generate error:", errorText);
      return buildLocalAnswer(question, SAFE_CONTEXT);
    }

    const data = await res.json();
    const text = data?.candidates?.[0]?.content?.parts?.[0]?.text?.trim() ?? "";
    return text || buildLocalAnswer(question, SAFE_CONTEXT);
  } catch (e: any) {
    console.warn("Gemini exception:", e?.message || e);
    return buildLocalAnswer(question, SAFE_CONTEXT);
  }
}
