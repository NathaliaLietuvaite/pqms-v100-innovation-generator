-- Create knowledge_base table for RAG-based retrieval
CREATE TABLE public.knowledge_base (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  version_key TEXT NOT NULL UNIQUE,
  title TEXT NOT NULL,
  summary TEXT NOT NULL,
  category TEXT NOT NULL DEFAULT 'core',
  keywords TEXT[] NOT NULL DEFAULT '{}',
  file_path TEXT,
  is_milestone BOOLEAN NOT NULL DEFAULT false,
  is_draft BOOLEAN NOT NULL DEFAULT false,
  sort_order INTEGER NOT NULL DEFAULT 0,
  search_vector tsvector,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Create indexes
CREATE INDEX idx_knowledge_base_search ON public.knowledge_base USING GIN(search_vector);
CREATE INDEX idx_knowledge_base_category ON public.knowledge_base(category);
CREATE INDEX idx_knowledge_base_version ON public.knowledge_base(version_key);

-- Trigger to update search_vector
CREATE OR REPLACE FUNCTION public.knowledge_base_search_update()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = public
AS $$
BEGIN
  NEW.search_vector :=
    setweight(to_tsvector('simple', coalesce(NEW.title, '')), 'A') ||
    setweight(to_tsvector('simple', coalesce(NEW.summary, '')), 'B') ||
    setweight(to_tsvector('simple', coalesce(array_to_string(NEW.keywords, ' '), '')), 'C');
  RETURN NEW;
END;
$$;

CREATE TRIGGER trg_knowledge_base_search
  BEFORE INSERT OR UPDATE ON public.knowledge_base
  FOR EACH ROW
  EXECUTE FUNCTION public.knowledge_base_search_update();

-- Enable RLS
ALTER TABLE public.knowledge_base ENABLE ROW LEVEL SECURITY;

-- Public read access for authenticated users
CREATE POLICY "Knowledge base is publicly readable"
  ON public.knowledge_base
  FOR SELECT
  TO authenticated
  USING (true);

-- Search function for RAG retrieval
CREATE OR REPLACE FUNCTION public.search_knowledge(
  search_query TEXT,
  match_limit INTEGER DEFAULT 10
)
RETURNS TABLE (
  id UUID,
  version_key TEXT,
  title TEXT,
  summary TEXT,
  category TEXT,
  keywords TEXT[],
  file_path TEXT,
  is_milestone BOOLEAN,
  is_draft BOOLEAN,
  rank REAL
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
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
$$;

-- Trigger for updated_at
CREATE TRIGGER update_knowledge_base_updated_at
  BEFORE UPDATE ON public.knowledge_base
  FOR EACH ROW
  EXECUTE FUNCTION public.update_updated_at_column();