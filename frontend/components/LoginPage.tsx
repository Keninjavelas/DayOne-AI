"use client";

import { FormEvent, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { apiRequest } from "../../lib/api";

type AuthToken = {
  access_token: string;
  token_type: string;
  username: string;
  organization: string;
  role: "admin" | "employee" | string;
  expires_at: string;
};

type JwtPayload = {
  sub?: string;
  username?: string;
  organization?: string;
  role?: string;
  exp?: number;
};

type LoginPageProps = {
  apiBaseUrl?: string;
};

function decodeJwt(token: string): JwtPayload | null {
  try {
    const payload = token.split(".")[1];
    if (!payload) return null;
    const base64 = payload.replace(/-/g, "+").replace(/_/g, "/");
    const padded = base64.padEnd(base64.length + ((4 - (base64.length % 4)) % 4), "=");
    const json = atob(padded);
    return JSON.parse(json) as JwtPayload;
  } catch {
    return null;
  }
}

export default function LoginPage({ apiBaseUrl }: LoginPageProps) {
  const router = useRouter();
  const apiRoot = useMemo(
    () => apiBaseUrl ?? process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000",
    [apiBaseUrl],
  );
  const defaultOrganization = process.env.NEXT_PUBLIC_DEMO_ORGANIZATION ?? "org_acme";

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [organization, setOrganization] = useState(defaultOrganization);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const data = await apiRequest<
        Partial<AuthToken> & { detail?: string },
        { username: string; password: string; organization: string }
      >({
        url: `${apiRoot}/auth/login`,
        method: "POST",
        data: { username, password, organization },
      });

      if (!data.access_token) {
        throw new Error("Login succeeded but no access token was returned.");
      }

      localStorage.setItem("dayone_token", data.access_token);
      const decoded = decodeJwt(data.access_token);
      const role = decoded?.role || data.role || "employee";

      localStorage.setItem(
        "dayone_profile",
        JSON.stringify({
          username: decoded?.username || data.username || username,
          organization: decoded?.organization || data.organization || organization,
          role,
        }),
      );

      router.replace(role === "admin" ? "/admin" : "/chat");
    } catch (err: any) {
      const message = err.response?.data?.detail || err.message || "Unable to log in";
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="login-root">
      <div className="login-card">
        <div className="monogram">D1</div>
        <h1 style={{ color: "white", textAlign: "center", margin: "0 0 0.5rem", fontSize: "1.5rem", fontWeight: 700 }}>Welcome Back</h1>
        <p style={{ color: "#94a3b8", textAlign: "center", margin: "0 0 2rem", fontSize: "0.9rem" }}>Log in to your organization dashboard</p>

        {error && <div className="error-box">{error}</div>}

        <form onSubmit={handleSubmit}>
          <div>
            <label className="login-label">Organization</label>
            <input
              className="login-input"
              type="text"
              placeholder="e.g. Acme Corp"
              value={organization}
              onChange={(e) => setOrganization(e.target.value)}
              required
            />
          </div>
          <div>
            <label className="login-label">Username</label>
            <input
              className="login-input"
              type="text"
              placeholder="Your username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
          </div>
          <div>
            <label className="login-label">Password</label>
            <input
              className="login-input"
              type="password"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>
          <button className="login-btn" type="submit" disabled={loading}>
            {loading ? "Logging in..." : "Continue"}
          </button>
        </form>

        <p style={{ color: "#64748b", textAlign: "center", marginTop: "1.5rem", fontSize: "0.85rem" }}>
          Don't have an organization?{" "}
          <button 
            onClick={() => router.push("/signup")}
            style={{ color: "#38bdf8", background: "none", border: "none", padding: 0, cursor: "pointer", fontWeight: 500 }}
          >
            Sign up
          </button>
        </p>
      </div>
    </div>
  );
}
