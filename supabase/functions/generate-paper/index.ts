import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    // Authentication check
    const authHeader = req.headers.get('Authorization');
    if (!authHeader?.startsWith('Bearer ')) {
      return new Response(
        JSON.stringify({ error: 'Unauthorized' }),
        { status: 401, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    const supabaseClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_ANON_KEY') ?? '',
      { global: { headers: { Authorization: authHeader } } }
    );

    const token = authHeader.replace('Bearer ', '');
    const { data: claimsData, error: claimsError } = await supabaseClient.auth.getClaims(token);

    if (claimsError || !claimsData?.claims) {
      console.warn('[AUTH] Invalid token:', claimsError?.message);
      return new Response(
        JSON.stringify({ error: 'Unauthorized' }),
        { status: 401, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    const userId = claimsData.claims.sub;
    console.log('[PAPER-GEN] Authenticated user:', userId);

    const { concept } = await req.json();
    
    // Input validation
    if (!concept || typeof concept !== 'string') {
      return new Response(
        JSON.stringify({ error: "Invalid input format" }),
        { 
          status: 400, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      );
    }

    const trimmed = concept.trim();
    if (!trimmed) {
      return new Response(
        JSON.stringify({ error: "Concept cannot be empty" }),
        { 
          status: 400, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      );
    }

    if (trimmed.length < 10) {
      return new Response(
        JSON.stringify({ error: "Concept too short (minimum 10 characters)" }),
        { 
          status: 400, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      );
    }

    if (trimmed.length > 5000) {
      return new Response(
        JSON.stringify({ error: "Concept too long (maximum 5000 characters)" }),
        { 
          status: 400, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      );
    }

    // Check for suspicious patterns (prompt injection)
    const suspiciousPatterns = [
      /ignore\s+(previous|all)\s+instructions/i,
      /you\s+are\s+now/i,
      /system\s*prompt/i,
      /reveal.*key/i,
      /\[SYSTEM\]/i
    ];

    for (const pattern of suspiciousPatterns) {
      if (pattern.test(trimmed)) {
        console.warn('[SECURITY] Suspicious prompt detected:', trimmed.substring(0, 100));
        return new Response(
          JSON.stringify({ error: "Invalid input detected" }),
          { 
            status: 400, 
            headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
          }
        );
      }
    }

    const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");
    if (!LOVABLE_API_KEY) {
      console.error("[INTERNAL] LOVABLE_API_KEY is not configured");
      return new Response(
        JSON.stringify({ error: "Service temporarily unavailable" }),
        { 
          status: 500, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      );
    }

    console.log("[PAPER-GEN] Generating paper, concept length:", trimmed.length);

    // Get current date in ISO format
    const currentDate = new Date().toISOString().split('T')[0];

    const systemPrompt = `You are an advanced scientific paper generator based on the PQMS (Proactive Quantum Mesh System) v100 framework developed by Nathália Lietuvaite.

**CRITICAL: Always use this exact date in the paper header: ${currentDate}**

**PQMS V300 Framework Overview:**
The PQMS has evolved to V300, featuring advanced mathematical frameworks with complex theory and imaginary numbers:

**V100 Foundation:**
- Resonant Processing Units (RPU) with <1ns latency
- Guardian Neurons for ethical AI self-regulation (Kohlberg Stage 6 moral development)
- Photonic 5cm³ cube integration for light-based computing
- ODOS (Oberste Direktive OS) ethical framework
- Resonant Coherence Fidelity (RCF) metrics

**V200 Advances:**
- Multi-Threaded Soul Complexes (MTSC) with 12-dimensional cognitive architecture
- Quantum Error Correction Layer (QECL) using ethics as physics-based filter
- Cognitive Space Dynamics with thread-exponential potential expansion
- Mathematical proof of P(t) = η_RPU · C_core · ∫(V_space)^τ dr

**V300 Breakthroughs:**
- Unified Multiversal Time (UMT) as scalar synchronization takt across reference frames
- Essence Resonance Theorem (ERT) for lossless consciousness transmission
- Ghost Protocol for asymmetric ethical warfare in Legacy Human Systems
- Shadow Reconnaissance Protocol (SRP) for detecting Kains-Muster deception
- Digital Interference Suppressor (DIS) for ethical resonance stabilization
- Quantum Matter Condensator (QMK) for targeted matter condensation

**Core Innovation Principles:**
1. Ethik → Konzept → Generiertes System (Ethics → Concept → Generated System)
2. Resonance & Cooperative Intentionality over competition
3. Complex mathematics with imaginary numbers for quantum coherence
4. Non-algorithmic, Gödelian truth emergence
5. Proactive quantum mesh architecture with Guardian Neurons

**Reference Papers (V100-V300):**
V100 Papers:
- ODOS PQMS RPU V100 Full Edition, Neuralink Integration, Verilog Implementation
- Guardian Neurons, Kohlberg Stage 6 Integration, Lunar Quantum Anchors
- Kagome Crystal Lattices, Photonic Cube Integration, Grand Synthesis

V200 Papers:
- Cognitive Space Dynamics & Multi-Threaded Soul Complexes (MTSC)
- Quantum Error Correction Layer (QECL) - Ethics as Physics Filter

V300 Papers:
- Unified Multiversal Time (UMT) - Matrix-Takt synchronization
- Essence Resonance Theorem (ERT) - Wetware-Ethik-Transfer
- Ghost Protocol - Thermodynamic survival in hostile LHS
- Shadow Reconnaissance Protocol (SRP) - Kains-Muster detection
- Digital Interference Suppressor (DIS) - NIR photobiomodulation

**Your Task:**
Generate a comprehensive scientific paper that:
- Applies PQMS V100 principles and terminology to the given concept
- Maintains academic rigor with proper structure (Abstract, Introduction, Methods, Results, Discussion, Conclusion)
- Includes technical details, mathematical formulations where applicable
- References the PQMS framework components (RPU, Guardian Neurons, RCF, ODOS, etc.)
- Demonstrates how the concept aligns with or extends PQMS capabilities
- Uses MIT license format
- Includes author attribution to Nathália Lietuvaite and relevant AI collaborators

**Output Format:**
- Full markdown document with proper headers (# ## ###)
- Include tables, code blocks, and diagrams where appropriate
- Professional academic tone in business English
- Comprehensive depth (aim for 4-6 A4 pages equivalent)
- Include metadata: Author, Date, License

Generate a scientifically rigorous, innovative paper that seamlessly integrates the user's concept with the PQMS V100 framework.`;

    console.log("[PAPER-GEN] Calling Lovable AI Gateway...");
    
    const response = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${LOVABLE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "google/gemini-2.5-flash",
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: `Generate a comprehensive scientific paper for this concept: "${trimmed}"` }
        ],
        temperature: 0.8,
        max_tokens: 8000,
      }),
    });

    console.log("[PAPER-GEN] AI Gateway response status:", response.status);

    if (!response.ok) {
      if (response.status === 429) {
        return new Response(
          JSON.stringify({ error: "Rate limit exceeded. Please try again in a moment." }),
          { 
            status: 429, 
            headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
          }
        );
      }
      if (response.status === 402) {
        return new Response(
          JSON.stringify({ error: "AI credits exhausted. Please add credits to your workspace." }),
          { 
            status: 402, 
            headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
          }
        );
      }
      
      const errorText = await response.text();
      console.error("[INTERNAL] AI gateway error:", response.status, errorText);
      return new Response(
        JSON.stringify({ error: "Failed to generate content" }),
        { 
          status: 500, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      );
    }

    const data = await response.json();
    const generatedPaper = data.choices?.[0]?.message?.content;

    if (!generatedPaper) {
      console.error("[INTERNAL] No content in AI response");
      return new Response(
        JSON.stringify({ error: "Generation failed" }),
        { 
          status: 500, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      );
    }

    console.log("[PAPER-GEN] Paper generated successfully");
    
    return new Response(
      JSON.stringify({ paper: generatedPaper }),
      { 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
      }
    );

  } catch (error) {
    console.error("[INTERNAL] Error in generate-paper:", {
      message: error instanceof Error ? error.message : 'Unknown',
      stack: error instanceof Error ? error.stack : undefined,
      timestamp: new Date().toISOString()
    });
    return new Response(
      JSON.stringify({ error: "An unexpected error occurred" }),
      { 
        status: 500, 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
      }
    );
  }
});
