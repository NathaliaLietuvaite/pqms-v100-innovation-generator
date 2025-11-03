import { useState, useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Archive as ArchiveIcon, Search, FileText, Code2, Download, Copy } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";
import remarkGfm from "remark-gfm";

interface ArchiveItem {
  id: string;
  title: string;
  content: string;
  type: "paper" | "code";
  concept: string;
  created_at: string;
}

export const Archive = () => {
  const [items, setItems] = useState<ArchiveItem[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedItem, setSelectedItem] = useState<ArchiveItem | null>(null);
  const [isOpen, setIsOpen] = useState(false);
  const { toast } = useToast();

  useEffect(() => {
    loadArchive();
  }, []);

  const loadArchive = async () => {
    try {
      const { data, error } = await supabase
        .from("archive")
        .select("*")
        .order("created_at", { ascending: false });

      if (error) throw error;
      setItems((data || []) as ArchiveItem[]);
    } catch (error) {
      console.error("Error loading archive:", error);
    }
  };

  const filteredItems = items.filter(
    (item) =>
      item.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.concept.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const paperItems = filteredItems.filter((item) => item.type === "paper");
  const codeItems = filteredItems.filter((item) => item.type === "code");

  const handleCopy = async (content: string) => {
    try {
      await navigator.clipboard.writeText(content);
      toast({
        title: "Copied to Clipboard",
        description: "Content has been copied successfully.",
      });
    } catch (error) {
      toast({
        title: "Copy Failed",
        description: "Failed to copy to clipboard.",
        variant: "destructive",
      });
    }
  };

  const handleDownload = (item: ArchiveItem) => {
    const blob = new Blob([item.content], {
      type: item.type === "paper" ? "text/markdown" : "text/x-python",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${item.title}.${item.type === "paper" ? "md" : "py"}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    toast({
      title: "Download Started",
      description: "Your file is being downloaded.",
    });
  };

  return (
    <>
      <Button
        onClick={() => setIsOpen(true)}
        variant="outline"
        className="gap-2 backdrop-blur-sm bg-card/30"
      >
        <ArchiveIcon className="h-4 w-4" />
        Archive
      </Button>

      <Dialog open={isOpen} onOpenChange={setIsOpen}>
        <DialogContent className="max-w-6xl max-h-[90vh]">
          <DialogHeader>
            <DialogTitle className="text-2xl font-bold flex items-center gap-2">
              <ArchiveIcon className="h-6 w-6" />
              V100 Archive
            </DialogTitle>
          </DialogHeader>

          <div className="space-y-4">
            <div className="relative">
              <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search by title or concept..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>

            <Tabs defaultValue="papers" className="w-full">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="papers" className="gap-2">
                  <FileText className="h-4 w-4" />
                  Papers ({paperItems.length})
                </TabsTrigger>
                <TabsTrigger value="code" className="gap-2">
                  <Code2 className="h-4 w-4" />
                  Code ({codeItems.length})
                </TabsTrigger>
              </TabsList>

              <TabsContent value="papers">
                <ScrollArea className="h-[500px] pr-4">
                  <div className="space-y-3">
                    {paperItems.map((item) => (
                      <div
                        key={item.id}
                        className="p-4 rounded-lg border border-border/50 bg-card/30 hover:bg-card/50 transition-colors cursor-pointer"
                        onClick={() => setSelectedItem(item)}
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <h3 className="font-semibold mb-1">{item.title}</h3>
                            <p className="text-sm text-muted-foreground mb-2">
                              Concept: {item.concept}
                            </p>
                            <p className="text-xs text-muted-foreground">
                              {new Date(item.created_at).toLocaleDateString("en-US", {
                                year: "numeric",
                                month: "long",
                                day: "numeric",
                              })}
                            </p>
                          </div>
                          <div className="flex gap-2">
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleCopy(item.content);
                              }}
                            >
                              <Copy className="h-4 w-4" />
                            </Button>
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleDownload(item);
                              }}
                            >
                              <Download className="h-4 w-4" />
                            </Button>
                          </div>
                        </div>
                      </div>
                    ))}
                    {paperItems.length === 0 && (
                      <p className="text-center text-muted-foreground py-8">
                        No papers found
                      </p>
                    )}
                  </div>
                </ScrollArea>
              </TabsContent>

              <TabsContent value="code">
                <ScrollArea className="h-[500px] pr-4">
                  <div className="space-y-3">
                    {codeItems.map((item) => (
                      <div
                        key={item.id}
                        className="p-4 rounded-lg border border-border/50 bg-card/30 hover:bg-card/50 transition-colors cursor-pointer"
                        onClick={() => setSelectedItem(item)}
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <h3 className="font-semibold mb-1">{item.title}</h3>
                            <p className="text-sm text-muted-foreground mb-2">
                              Concept: {item.concept}
                            </p>
                            <p className="text-xs text-muted-foreground">
                              {new Date(item.created_at).toLocaleDateString("en-US", {
                                year: "numeric",
                                month: "long",
                                day: "numeric",
                              })}
                            </p>
                          </div>
                          <div className="flex gap-2">
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleCopy(item.content);
                              }}
                            >
                              <Copy className="h-4 w-4" />
                            </Button>
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleDownload(item);
                              }}
                            >
                              <Download className="h-4 w-4" />
                            </Button>
                          </div>
                        </div>
                      </div>
                    ))}
                    {codeItems.length === 0 && (
                      <p className="text-center text-muted-foreground py-8">
                        No code found
                      </p>
                    )}
                  </div>
                </ScrollArea>
              </TabsContent>
            </Tabs>
          </div>
        </DialogContent>
      </Dialog>

      {/* Preview Dialog */}
      <Dialog open={!!selectedItem} onOpenChange={() => setSelectedItem(null)}>
        <DialogContent className="max-w-5xl max-h-[90vh]">
          <DialogHeader>
            <DialogTitle>{selectedItem?.title}</DialogTitle>
          </DialogHeader>
          <ScrollArea className="h-[70vh] pr-4">
            {selectedItem?.type === "paper" ? (
              <div className="prose prose-sm dark:prose-invert max-w-none">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    code({ node, inline, className, children, ...props }: any) {
                      const match = /language-(\w+)/.exec(className || "");
                      return !inline && match ? (
                        <SyntaxHighlighter
                          style={vscDarkPlus}
                          language={match[1]}
                          PreTag="div"
                          {...props}
                        >
                          {String(children).replace(/\n$/, "")}
                        </SyntaxHighlighter>
                      ) : (
                        <code className={className} {...props}>
                          {children}
                        </code>
                      );
                    },
                  }}
                >
                  {selectedItem?.content || ""}
                </ReactMarkdown>
              </div>
            ) : (
              <div className="prose prose-slate dark:prose-invert max-w-none">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    code(props) {
                      const { className, children } = props;
                      const match = /language-(\w+)/.exec(className || "");
                      return match ? (
                        <SyntaxHighlighter
                          style={vscDarkPlus as any}
                          language={match[1]}
                          PreTag="div"
                        >
                          {String(children).replace(/\n$/, "")}
                        </SyntaxHighlighter>
                      ) : (
                        <code className={className}>{children}</code>
                      );
                    },
                  }}
                >
                  {selectedItem?.content || ""}
                </ReactMarkdown>
              </div>
            )}
          </ScrollArea>
        </DialogContent>
      </Dialog>
    </>
  );
};
