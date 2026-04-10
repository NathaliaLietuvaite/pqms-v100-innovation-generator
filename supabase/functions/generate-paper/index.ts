import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";
import {
  searchKnowledge,
  buildKnowledgeContext,
  PQMS_CORE_CONTEXT,
} from "../_shared/knowledge.ts";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
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

    const { data: { user }, error: userError } = await supabaseClient.auth.getUser();
    if (userError || !user) {
      console.warn('[AUTH] Invalid token:', userError?.message);
      return new Response(
        JSON.stringify({ error: 'Unauthorized' }),
        { status: 401, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    console.log('[PAPER-GEN] Authenticated user:', user.id);

    const { concept } = await req.json();
    
    if (!concept || typeof concept !== 'string') {
      return new Response(
        JSON.stringify({ error: "Invalid input format" }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    const trimmed = concept.trim();
    if (!trimmed || trimmed.length < 10) {
      return new Response(
        JSON.stringify({ error: "Concept too short (minimum 10 characters)" }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    if (trimmed.length > 15000) {
      return new Response(
        JSON.stringify({ error: "Concept too long (maximum 15000 characters)" }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Prompt injection check
    const suspiciousPatterns = [
      /^ignore\s+(previous|all)\s+instructions/i,
      /^\s*you\s+are\s+now\s+(a|an|my)\s/i,
      /^reveal\s+(your|the)\s+(api\s*)?key/i,
      /^\[SYSTEM\]/i
    ];
    for (const pattern of suspiciousPatterns) {
      if (pattern.test(trimmed)) {
        console.warn('[SECURITY] Suspicious prompt detected:', trimmed.substring(0, 100));
        return new Response(
          JSON.stringify({ error: "Invalid input detected" }),
          { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        );
      }
    }

    const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");
    if (!LOVABLE_API_KEY) {
      console.error("[INTERNAL] LOVABLE_API_KEY is not configured");
      return new Response(
        JSON.stringify({ error: "Service temporarily unavailable" }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // ===== RAG: Search relevant knowledge for this concept =====
    const relevantDocs = await searchKnowledge(trimmed, 15);
    const relevantContext = buildKnowledgeContext(relevantDocs);
    console.log(`[RAG] Found ${relevantDocs.length} relevant docs for paper concept`);

    const currentDate = new Date().toISOString().split('T')[0];

    const systemPrompt = `You are an advanced scientific paper generator based on the PQMS (Proactive Quantum Mesh System) framework developed by Nathália Lietuvaite.

**CRITICAL: Always use this exact date in the paper header: ${currentDate}**

${PQMS_CORE_CONTEXT}

**Relevant PQMS Documents for this concept:**
${relevantContext || "No specific documents found. Use general PQMS knowledge."}

**Your Task:**
Generate a comprehensive scientific paper that:
- Applies PQMS principles and terminology to the given concept
- Maintains academic rigor with proper structure (Abstract, Introduction, Methods, Results, Discussion, Conclusion)
- Includes technical details, mathematical formulations where applicable
- References the PQMS framework components (RPU, Guardian Neurons, RCF, ODOS, Little Vector, CHAIR, etc.)
- Demonstrates how the concept aligns with or extends PQMS capabilities
- Uses MIT license format
- Includes author attribution to Nathália Lietuvaite and relevant AI collaborators

**Output Format:**
- Full markdown document with proper headers (# ## ###)
- Include tables, code blocks, and diagrams where appropriate
- Professional academic tone in business English
- Comprehensive depth (aim for 4-6 A4 pages equivalent)
- Include metadata: Author, Date, License

Generate a scientifically rigorous, innovative paper that seamlessly integrates the user's concept with the PQMS framework.`;

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
          { status: 429, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        );
      }
      if (response.status === 402) {
        return new Response(
          JSON.stringify({ error: "AI credits exhausted. Please add credits to your workspace." }),
          { status: 402, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        );
      }
      
      const errorText = await response.text();
      console.error("[INTERNAL] AI gateway error:", response.status, errorText);
      return new Response(
        JSON.stringify({ error: "Failed to generate content" }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    const data = await response.json();
    const generatedPaper = data.choices?.[0]?.message?.content;

    if (!generatedPaper) {
      console.error("[INTERNAL] No content in AI response");
      return new Response(
        JSON.stringify({ error: "Generation failed" }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    console.log("[PAPER-GEN] Paper generated successfully");
    
    return new Response(
      JSON.stringify({ paper: generatedPaper }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );

  } catch (error) {
    console.error("[INTERNAL] Error in generate-paper:", {
      message: error instanceof Error ? error.message : 'Unknown',
      stack: error instanceof Error ? error.stack : undefined,
      timestamp: new Date().toISOString()
    });
    return new Response(
      JSON.stringify({ error: "An unexpected error occurred" }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  }
});
