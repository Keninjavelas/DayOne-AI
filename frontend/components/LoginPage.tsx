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

  return (
    <div className="login-root">
      <div className="auth-shell">
        <section className="auth-hero glass-card">
          <div className="monogram monogram-large">D1</div>
          <p className="eyebrow">Secure HR intelligence</p>
          <h1 className="auth-hero-title">Welcome back to DayOne AI</h1>
          <p className="auth-hero-copy">
            Sign in to review policies, answer employee questions, and keep every response grounded in your organization&apos;s documents.
          </p>

          <div className="feature-stack">
            <div className="feature-pill">Grounded answers from approved files</div>
            <div className="feature-pill">Multi-tenant organization isolation</div>
            <div className="feature-pill">Confidence and source details built in</div>
          </div>
        </section>

        <section className="auth-panel glass-card">
          <div className="auth-panel-header">
            <p className="eyebrow">Sign in</p>
            <h2 className="auth-panel-title">Access your workspace</h2>
          </div>

          {error && <div className="error-box">{error}</div>}

          <form onSubmit={handleSubmit} className="auth-form">
            <div className="field-group">
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
            <div className="field-group">
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
            <div className="field-group">
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

          <p className="auth-footer">
            Don&apos;t have an organization? {" "}
            <button
              onClick={() => router.push("/signup")}
              className="link-btn"
              type="button"
            >
              Sign up
            </button>
          </p>
        </section>
      </div>
    </div>
  );
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
