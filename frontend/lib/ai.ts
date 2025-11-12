import { pipeline } from "@xenova/transformers";

// ====================
// 🔹 Embeddings locales (384 dimensiones)
// ====================
let embedder: any = null;

export async function embedWithGemini(text: string): Promise<number[]> {
  if (!embedder) {
    console.log("🧠 Cargando modelo local de embeddings (all-MiniLM-L6-v2)...");
    embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
  }

  const output = await embedder(text, { pooling: "mean", normalize: true });
  return Array.from(output.data);
}

// ====================
// 🔹 Generación de respuesta con Gemini
// ====================
export async function askGemini(question: string, context: string): Promise<string> {
  const GEMINI_KEY = process.env.GEMINI_API_KEY;

  if (!GEMINI_KEY) {
    console.warn("⚠️ No hay GEMINI_API_KEY, respondiendo localmente (modo offline).");
    return `No tengo acceso a Gemini actualmente, pero puedo ayudarte con el contexto disponible:\n\n${context.slice(0, 500)}...`;
  }

  const res = await fetch(
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=" + GEMINI_KEY,
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
Responde **solo** con información del contexto proporcionado.
Si no tienes datos suficientes, responde "No tengo información suficiente".
Incluye referencias al final en formato [doc:nombre].

Pregunta: ${question}

Contexto:
${context}`,
              },
            ],
          },
        ],
        generationConfig: {
          temperature: 0.3,
          maxOutputTokens: 512,
        },
      }),
    }
  );

  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`Gemini generate error ${res.status}: ${errorText}`);
  }

  const data = await res.json();
  const text = data?.candidates?.[0]?.content?.parts?.[0]?.text ?? "";
  return text.trim();
}
