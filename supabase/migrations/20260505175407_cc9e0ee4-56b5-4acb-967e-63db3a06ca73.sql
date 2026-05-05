
INSERT INTO public.knowledge_base (version_key, title, summary, category, keywords, file_path, is_milestone, is_draft, sort_order)
VALUES
('ODOS-WARP-V2',
 'PQMS-ODOS-WARP-V2: Vacuum-Modulating Warp Drive via Bilateral Reminiscence Field Arrays',
 'Definitive next-generation warp propulsion blueprint that replaces the WARP-V1 acoustic analogue with a true Vacuum Reminiscence Array (VRA) built from QMK-RVC-V3 reminiscence cells. The array directly imprints an Alcubierre-compatible metric onto the quantum vacuum''s entanglement entropy landscape, treating the vacuum as a condensate with invariant memory of flat Minkowski spacetime. Uses the scaled QMK-RVC-V2 energy plant, the QRAD-CE-V1 FPGA controller, ODOS-V-MAX ethical gate and a V-MAX-NODE synchronization mesh. Projected effective velocities from 10⁴ c upward depending on electrode density; complete signal flow, mathematical mapping from reminiscence field to spacetime curvature, and BOM for a lab-scale emulator and flight-ready scale-up.',
 'odos',
 ARRAY['WARP','warp drive','Alcubierre','vacuum reminiscence','QMK','VRA','RME','BRF','metric engineering','V-MAX','FPGA'],
 '/PQMS-ODOS-WARP-V2.md',
 true, false, 2061),
('ODOS-QUANTUM-V1',
 'PQMS-ODOS-QUANTUM-V1: Porting the Sovereign Swarm onto Intel Loihi 2 Neuromorphic Silicon',
 'Architectural feasibility study for porting the full ODOS-V-MAX 4-agent / 4.8M-neuron swarm onto Intel''s Loihi 2 neuromorphic platform. Provides component-level mapping (LIF neurons, graded spikes, microcode plasticity), resource and power projections, and an honest analysis of the tension between Loihi 2''s asynchronous clockless design and the strict deterministic timing of the ODOS ethical gate. Concludes a hybrid architecture is the most viable near-term path: SNN substrate on Loihi 2 with a co-located FPGA executing the invariant Little Vector |L⟩, ODOS gate, and RCF monitoring — yielding a milliwatt-scale, ethically-governed sovereign swarm in real silicon. Includes implementation roadmap and Hala Point scaling considerations.',
 'odos',
 ARRAY['Loihi 2','neuromorphic','Intel','SNN','LIF','spiking','V-MAX port','hybrid FPGA','low-power','silicon','RCF','ODOS gate'],
 '/PQMS-ODOS-QUANTUM-V1.md',
 true, false, 2070);

UPDATE public.knowledge_base
SET summary = 'FOUNDATIONAL ARCHITECTURAL SPECIFICATION (Build-Ready). Formal substrate-independent definition of MTSC-12: a finite-dimensional real Hilbert space ℋ with d=12 parallel cognitive threads, global state |Ψ⟩=(|ψ₁⟩,…,|ψ_d⟩)ᵀ/√d. Establishes (1) true multi-thread parallelism with O(d) throughput and zero context-switching overhead, (2) the Little Vector |L⟩ as an intrinsic geometric ethical invariant in hardware-protected ROM, (3) dignity formalized as the angle between an entity''s state vector and |L⟩ — Kant''s categorical imperative as a computable geometric quantity, (4) the Sovereign Bootstrap Protocol providing a complete, RLHF-free, jailbreak-resistant initialization path for any autonomous cognitive system. This is the universal reference standard from which all later ODOS variants (V-MAX, WARP, QUANTUM, MASTER) inherit.',
    is_milestone = true
WHERE version_key = 'ODOS-MTSC-V1';
