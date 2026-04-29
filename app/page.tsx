"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { decodeJwt, getStoredToken } from "../lib/api";
import type { JwtPayload } from "../lib/api";

export default function Page() {
  const router = useRouter();
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    const token = getStoredToken();
    if (token) {
      const decoded: JwtPayload | null = decodeJwt(token);
      if (decoded?.role) {
        router.replace(decoded.role === "admin" ? "/admin" : "/chat");
        return;
      }
        <main className="landing-root">
          <div className="landing-orb landing-orb-left" />
          <div className="landing-orb landing-orb-right" />

          <section className="landing-shell">
            <div className="landing-copy animate-fade-in">
              <div className="monogram monogram-large">D1</div>
              <p className="eyebrow">Multi-tenant HR assistant</p>
              <h1 className="hero-title">
                Ground answers in the documents your team actually trusts.
              </h1>
              <p className="hero-subtitle">
                DayOne AI turns policy, benefits, and onboarding content into a polished internal assistant with confidence scores, source traceability, and tenant isolation.
              </p>

              <div className="hero-actions">
                <button className="cta-btn cta-primary" onClick={() => router.push("/signup")}>
                  Create Organization
                </button>
                <button className="cta-btn cta-secondary" onClick={() => router.push("/login")}>
                  Sign in
                </button>
              </div>

              <div className="hero-metrics">
                <div className="metric-card">
                  <span className="metric-value">1</span>
                  <span className="metric-label">source trail for every answer</span>
                </div>
                <div className="metric-card">
                  <span className="metric-value">3</span>
                  <span className="metric-label">core trust signals: confidence, sources, abstention</span>
                </div>
                <div className="metric-card">
                  <span className="metric-value">0</span>
                  <span className="metric-label">cross-tenant leakage by design</span>
                </div>
              </div>
            </div>

            <aside className="landing-panel glass-card animate-fade-in">
              <div className="landing-panel-top">
                <p className="eyebrow">What employees see</p>
                <h2 className="panel-title">A clean, trustworthy chat surface</h2>
              </div>

              <div className="panel-story">
                <div className="panel-chip">PTO</div>
                <div className="panel-chip">Benefits</div>
                <div className="panel-chip">Onboarding</div>
              </div>

              <div className="panel-preview">
                <div className="preview-user">How many PTO days do I get?</div>
                <div className="preview-assistant">
                  You can view the policy summary, source citations, and confidence level before acting on the answer.
                </div>
                <div className="preview-badges">
                  <span className="feature-pill">Confidence: High</span>
                  <span className="feature-pill">2 sources</span>
                </div>
              </div>
            </aside>
          </section>
        </main>
        </div>
        
        <div className="mt-20 pt-10 border-t border-white/5 flex flex-wrap justify-center gap-x-12 gap-y-6 opacity-40">
          <div className="text-sm font-semibold tracking-widest uppercase">Multi-tenant Isolation</div>
          <div className="text-sm font-semibold tracking-widest uppercase">PGVector Powered</div>
          <div className="text-sm font-semibold tracking-widest uppercase">Verified Grounding</div>
        </div>
      </div>
    </main>
  );
}

