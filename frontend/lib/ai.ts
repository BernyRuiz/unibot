// frontend/lib/ai.ts
const GEMINI_KEY = process.env.GEMINI_API_KEY!;

// Embeddings con Gemini (modelo de embeddings)
export async function embedWithGemini(text: string): Promise<number[]> {
  const res = await fetch(
    "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedText?key=" + GEMINI_KEY,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    }
  );
  if (!res.ok) throw new Error("Gemini embeddings error");
  const data = await res.json();
  // estructura: data.embedding.values => number[]
  return data.embedding?.values ?? data?.data?.[0]?.embedding?.values ?? [];
}

// Generación con contexto (modelo rápido/costo-eficiente)
export async function askGemini(question: string, context: string): Promise<string> {
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
                text:
`Eres un asistente para alumnos. Responde SOLO con información del contexto.
Cita los fragmentos relevantes así: [doc:{docName}].
Si falta información, admite la limitación y sugiere revisar con un humano.

Pregunta: ${question}

Contexto:
${context}`
              }
            ]
          }
        ],
        generationConfig: {
          temperature: 0.2,
          maxOutputTokens: 512
        }
      })
    }
  );
  if (!res.ok) throw new Error("Gemini generate error");
  const data = await res.json();
  const text = data?.candidates?.[0]?.content?.parts?.[0]?.text ?? "";
  return text;
}
