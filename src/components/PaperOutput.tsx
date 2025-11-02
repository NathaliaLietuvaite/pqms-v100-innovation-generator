import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Copy, Download, Check } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";

interface PaperOutputProps {
  paper: string;
}

export const PaperOutput = ({ paper }: PaperOutputProps) => {
  const [copied, setCopied] = useState(false);
  const { toast } = useToast();

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(paper);
      setCopied(true);
      toast({
        title: "Copied to Clipboard",
        description: "The scientific paper has been copied to your clipboard.",
      });
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      toast({
        title: "Copy Failed",
        description: "Failed to copy to clipboard. Please try again.",
        variant: "destructive",
      });
    }
  };

  const handleDownloadMarkdown = () => {
    const blob = new Blob([paper], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `V100_Scientific_Paper_${new Date().toISOString().split('T')[0]}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    toast({
      title: "Download Started",
      description: "Your scientific paper is being downloaded as Markdown.",
    });
  };

  return (
    <div className="w-full max-w-5xl mx-auto space-y-6">
      <div className="flex gap-3 justify-end">
        <Button
          onClick={handleCopy}
          variant="outline"
          className="bg-background/50 backdrop-blur-sm hover:bg-background/80 transition-all"
        >
          {copied ? (
            <>
              <Check className="mr-2 h-4 w-4 text-green-500" />
              Copied
            </>
          ) : (
            <>
              <Copy className="mr-2 h-4 w-4" />
              Copy Markdown
            </>
          )}
        </Button>
        <Button
          onClick={handleDownloadMarkdown}
          variant="outline"
          className="bg-background/50 backdrop-blur-sm hover:bg-background/80 transition-all"
        >
          <Download className="mr-2 h-4 w-4" />
          Download as .md
        </Button>
      </div>

      <div className="backdrop-blur-sm bg-card/30 p-8 rounded-xl border border-border/50 shadow-lg">
        <div className="prose prose-sm sm:prose-base lg:prose-lg dark:prose-invert max-w-none">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              code({ node, inline, className, children, ...props }: any) {
                const match = /language-(\w+)/.exec(className || '');
                return !inline && match ? (
                  <SyntaxHighlighter
                    style={vscDarkPlus}
                    language={match[1]}
                    PreTag="div"
                    {...props}
                  >
                    {String(children).replace(/\n$/, '')}
                  </SyntaxHighlighter>
                ) : (
                  <code className={className} {...props}>
                    {children}
                  </code>
                );
              },
            }}
          >
            {paper}
          </ReactMarkdown>
        </div>
      </div>
    </div>
  );
};
