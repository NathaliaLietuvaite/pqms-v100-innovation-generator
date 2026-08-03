CREATE OR REPLACE FUNCTION public.search_knowledge(search_query text, match_limit integer DEFAULT 10)
 RETURNS TABLE(id uuid, version_key text, title text, summary text, category text, keywords text[], file_path text, is_milestone boolean, is_draft boolean, rank real)
 LANGUAGE plpgsql
 STABLE SECURITY INVOKER
 SET search_path TO 'public'
AS $function$
BEGIN
  RETURN QUERY
  SELECT
    kb.id,
    kb.version_key,
    kb.title,
    kb.summary,
    kb.category,
    kb.keywords,
    kb.file_path,
    kb.is_milestone,
    kb.is_draft,
    ts_rank(kb.search_vector, plainto_tsquery('simple', search_query)) AS rank
  FROM public.knowledge_base kb
  WHERE kb.search_vector @@ plainto_tsquery('simple', search_query)
  ORDER BY rank DESC
  LIMIT match_limit;
END;
$function$;

REVOKE ALL ON FUNCTION public.search_knowledge(text, integer) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.search_knowledge(text, integer) TO authenticated, service_role;