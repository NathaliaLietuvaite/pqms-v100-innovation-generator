DROP POLICY IF EXISTS "Knowledge base is publicly readable by anon" ON public.knowledge_base;
REVOKE SELECT ON public.knowledge_base FROM anon;