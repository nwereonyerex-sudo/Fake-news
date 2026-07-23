// Thin Vercel proxy for GET /health - see api/predict.js for why this
// forwards to Render instead of running inference locally.

export const config = { maxDuration: 30 };

export default async function handler(req, res) {
  const backendUrl = process.env.BACKEND_URL;
  if (!backendUrl) {
    res.status(500).json({ error: "BACKEND_URL is not configured" });
    return;
  }

  try {
    const upstream = await fetch(`${backendUrl.replace(/\/$/, "")}/health`);
    const data = await upstream.json();
    res.status(upstream.status).json(data);
  } catch (err) {
    res.status(502).json({ error: "Backend unreachable", detail: String(err) });
  }
}
