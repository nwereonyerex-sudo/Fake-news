// Thin Vercel proxy for POST /predict.
//
// Real inference (torch + the LSTM model) runs on Render, not here - the
// full dependency stack is ~700MB, well over Vercel's 250MB function limit.
// This function just forwards the request body to the Render-hosted API.
// Set BACKEND_URL in the Vercel project's environment variables to the
// deployed Render service's base URL (e.g. https://fake-news-detector-api.onrender.com).

export const config = { maxDuration: 60 };

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  const backendUrl = process.env.BACKEND_URL;
  if (!backendUrl) {
    res.status(500).json({ error: "BACKEND_URL is not configured" });
    return;
  }

  // Vercel lazily parses req.body on first access - if the client sent
  // malformed JSON, accessing it here throws. Catch that separately from
  // the fetch below so a client typo reports as 400, not a false "backend
  // unreachable" 502.
  let body;
  try {
    body = JSON.stringify(req.body);
  } catch (err) {
    res.status(400).json({ error: "Invalid JSON in request body", detail: String(err) });
    return;
  }

  try {
    const upstream = await fetch(`${backendUrl.replace(/\/$/, "")}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
    });
    const data = await upstream.json();
    res.status(upstream.status).json(data);
  } catch (err) {
    res.status(502).json({ error: "Backend unreachable", detail: String(err) });
  }
}
