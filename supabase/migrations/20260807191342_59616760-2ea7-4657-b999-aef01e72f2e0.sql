DROP POLICY IF EXISTS "Users can view their own conversations" ON public.creator_conversations;
DROP POLICY IF EXISTS "Users can insert their own conversations" ON public.creator_conversations;
DROP POLICY IF EXISTS "Users can update their own conversations" ON public.creator_conversations;

CREATE POLICY "Users can view their own conversations"
ON public.creator_conversations
FOR SELECT
TO authenticated
USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own conversations"
ON public.creator_conversations
FOR INSERT
TO authenticated
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own conversations"
ON public.creator_conversations
FOR UPDATE
TO authenticated
USING (auth.uid() = user_id)
WITH CHECK (auth.uid() = user_id);

REVOKE ALL ON public.creator_conversations FROM anon;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.creator_conversations TO authenticated;
GRANT ALL ON public.creator_conversations TO service_role;