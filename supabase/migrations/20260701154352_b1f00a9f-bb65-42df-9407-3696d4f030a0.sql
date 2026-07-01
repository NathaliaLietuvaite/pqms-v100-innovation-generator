
-- 1. Lock down SECURITY DEFINER search function: only service_role should call it.
REVOKE ALL ON FUNCTION public.search_knowledge(text, integer) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.search_knowledge(text, integer) FROM anon;
REVOKE ALL ON FUNCTION public.search_knowledge(text, integer) FROM authenticated;
GRANT EXECUTE ON FUNCTION public.search_knowledge(text, integer) TO service_role;

-- 2. Add DELETE policy for creator_conversations so users can manage their own data.
CREATE POLICY "Users can delete their own conversations"
  ON public.creator_conversations
  FOR DELETE
  TO authenticated
  USING (auth.uid() = user_id);

-- 3. Public knowledge base: allow anonymous read (matches demo-mode intent).
GRANT SELECT ON public.knowledge_base TO anon;

CREATE POLICY "Knowledge base is publicly readable by anon"
  ON public.knowledge_base
  FOR SELECT
  TO anon
  USING (true);
