"use client";

import { useState } from "react";

export default function Home() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);

  const ask = async () => {
    if (!question.trim()) return;
    setLoading(true);
    setAnswer("");
    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setAnswer(data.answer || "(sin respuesta)");
    } catch (err: any) {
      setAnswer("Error al consultar: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="flex flex-col items-center justify-center min-h-screen bg-gray-50 p-6">
      <div className="max-w-2xl w-full bg-white rounded-2xl shadow-lg p-8">
        <h1 className="text-2xl font-semibold mb-6 text-center">🤖 Unibot (RAG)</h1>

        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Escribe tu pregunta..."
          className="w-full p-3 border rounded-lg mb-4 resize-none h-32 focus:outline-none focus:ring-2 focus:ring-blue-500"
        />

        <button
          onClick={ask}
          disabled={loading}
          className="w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 transition"
        >
          {loading ? "Consultando..." : "Preguntar"}
        </button>

        {answer && (
          <div className="mt-6 p-4 border rounded-lg bg-gray-100 whitespace-pre-wrap">
            <strong>Respuesta:</strong>
            <p className="mt-2">{answer}</p>
          </div>
        )}
      </div>
    </main>
  );
}
