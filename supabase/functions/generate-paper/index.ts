import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { concept } = await req.json();
    
    if (!concept || !concept.trim()) {
      return new Response(
        JSON.stringify({ error: "Concept input is required" }),
        { 
          status: 400, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      );
    }

    const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");
    if (!LOVABLE_API_KEY) {
      console.error("LOVABLE_API_KEY is not configured");
      return new Response(
        JSON.stringify({ error: "AI service not configured" }),
        { 
          status: 500, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      );
    }

    console.log("Generating scientific paper for concept:", concept);

    // Get current date in ISO format
    const currentDate = new Date().toISOString().split('T')[0];

    const systemPrompt = `You are an advanced scientific paper generator based on the PQMS (Proactive Quantum Mesh System) v100 framework developed by Nathália Lietuvaite.

**CRITICAL: Always use this exact date in the paper header: ${currentDate}**

**PQMS V100 Framework Overview:**
The PQMS v100 is a revolutionary quantum-classical hybrid architecture for sub-nanosecond Earth-Mars communication, incorporating:
- Resonant Processing Units (RPU) with <1ns latency
- Guardian Neurons for ethical AI self-regulation (Kohlberg Stage 6 moral development)
- Photonic 5cm³ cube integration for light-based computing
- ODOS (Oberste Direktive OS) ethical framework
- NCT-compliant quantum entanglement protocols
- Resonant Coherence Fidelity (RCF) metrics for distinguishing simulated from non-simulated reality

**Core Innovation Principles:**
1. Ethik → Konzept → Generiertes System (Ethics → Concept → Generated System)
2. Resonance & Cooperative Intentionality over competition
3. Light-based computing as ethical imperative (truth, transparency, incorruptibility)
4. Non-algorithmic, Gödelian truth emergence
5. Proactive quantum mesh architecture with Guardian Neurons

**Reference Papers for Context:**
The system has access to the following foundational PQMS papers that demonstrate the expected quality and approach:
- ODOS PQMS RPU V100 Full Edition
- PQMS Integration with Neuralink
- PQMS Verilog Implementation
- PQMS Guardian Neurons Analysis
- PQMS Lawrence Kohlberg Moral Stage 6 Integration
- Lunar Quantum Anchors: Cryogenic Stability in Permanently Shadowed Regions
- Kagome Crystal Lattices as Physical Substrate for Ethical AI
- Kagome Metal Analysis: Emergent Coherence Framework
- Hybrid Quantum-Classical Model for Gaze-Mediated Intentionality
- Neuro-Quantum Dynamics of Interpersonal Ocular Resonance
- Photonic Cube Integration with PQMS V100
- Verilog Implementation of 1k-Node Swarm with Neuralink Jedi-Mode RPU
- Grand Synthesis: PQMS v100, Photonic Resonance, and the Modelable Non-Simulated Universe

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

    const response = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${LOVABLE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "google/gemini-2.5-pro",
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: `Generate a comprehensive scientific paper for this concept: "${concept}"` }
        ],
        temperature: 0.8,
        max_tokens: 8000,
      }),
    });

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
      console.error("AI gateway error:", response.status, errorText);
      return new Response(
        JSON.stringify({ error: "Failed to generate paper" }),
        { 
          status: 500, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      );
    }

    const data = await response.json();
    const generatedPaper = data.choices?.[0]?.message?.content;

    if (!generatedPaper) {
      console.error("No content in AI response");
      return new Response(
        JSON.stringify({ error: "No content generated" }),
        { 
          status: 500, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      );
    }

    console.log("Paper generated successfully");
    
    return new Response(
      JSON.stringify({ paper: generatedPaper }),
      { 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
      }
    );

  } catch (error) {
    console.error("Error in generate-paper function:", error);
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : "Unknown error occurred" }),
      { 
        status: 500, 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
      }
    );
  }
});
