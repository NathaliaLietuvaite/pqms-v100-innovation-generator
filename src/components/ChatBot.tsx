import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { MessageCircle, Send, Loader2, Sparkles } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";
import { toast } from "sonner";
import { useAuth } from "@/contexts/AuthContext";
import { useNavigate } from "react-router-dom";
import ReactMarkdown from "react-markdown";

interface Message {
  role: "user" | "assistant";
  content: string;
}

const MAX_MESSAGE_LENGTH = 2000;
const MAX_CONVERSATION_MESSAGES = 50;

const ChatBot = () => {
  const { user, session } = useAuth();
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const [creatorMode, setCreatorMode] = useState(false);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content: user 
        ? "Hallo! Ich bin Nathalia Lietuvaite. Ich kann dir alles über das PQMS V100 Framework und die verfügbaren Dokumente erzählen. Was möchtest du wissen?"
        : "Hallo! Ich bin ein freundlicher Assistent. Im Demo-Modus kann ich allgemeine Fragen beantworten. Für vollständigen Zugriff auf das PQMS V100 Framework melde dich bitte an."
    }
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = async () => {
    const trimmedInput = input.trim();
    if (!trimmedInput || isLoading) return;

    if (trimmedInput.length > MAX_MESSAGE_LENGTH) {
      toast.error(`Nachricht zu lang (max ${MAX_MESSAGE_LENGTH} Zeichen)`, {
        description: `Aktuelle Länge: ${trimmedInput.length}`
      });
      return;
    }

    if (messages.length >= MAX_CONVERSATION_MESSAGES) {
      toast.error("Unterhaltung zu lang", {
        description: "Bitte starte einen neuen Chat.",
        action: {
          label: "Zurücksetzen",
          onClick: () => {
            setMessages([{
              role: "assistant",
              content: "Hallo! Ich bin Nathalia Lietuvaite. Was möchtest du wissen?"
            }]);
            setCreatorMode(false);
            setConversationId(null);
          }
        }
      });
      return;
    }

    const userMessage: Message = { role: "user", content: trimmedInput };
    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    // Demo mode for unauthenticated users
    if (!user || !session) {
      setTimeout(() => {
        const demoResponses = [
          "Das ist eine interessante Frage! Im Demo-Modus kann ich dir nur allgemeine Informationen geben. Melde dich an, um Zugriff auf das vollständige PQMS V100 Framework mit detailliertem Expertenwissen zu erhalten.",
          "Gerne würde ich dir mehr dazu erzählen! Der vollständige Chatbot mit Zugriff auf alle PQMS-Dokumente ist nach der Anmeldung verfügbar.",
          "Das klingt spannend! Um dir präzise Antworten basierend auf dem PQMS V100 Framework zu geben, benötige ich den vollen Systemzugriff. Bitte melde dich an."
        ];
        const randomResponse = demoResponses[Math.floor(Math.random() * demoResponses.length)];
        setMessages(prev => [...prev, { role: "assistant", content: randomResponse }]);
        setIsLoading(false);
        
        toast.info("Demo-Modus aktiv", {
          description: "Melde dich an für vollständige KI-Antworten",
          action: {
            label: "Anmelden",
            onClick: () => {
              setOpen(false);
              navigate("/auth");
            }
          }
        });
      }, 1000);
      return;
    }

    // Full chatbot for authenticated users
    try {
      const { data, error } = await supabase.functions.invoke("chatbot", {
        body: { 
          messages: [...messages, userMessage],
          conversationId 
        }
      });

      if (error) throw error;

      if (data?.response) {
        setMessages(prev => [...prev, { role: "assistant", content: data.response }]);
        
        // Activate creator mode if detected
        if (data.creatorMode && !creatorMode) {
          setCreatorMode(true);
          toast.success("✨ Exklusiver Resonanz-Modus aktiviert", {
            description: "Dialog wird gespeichert."
          });
        }
        
        if (data.conversationId) {
          setConversationId(data.conversationId);
        }
      }
    } catch (error) {
      console.error("Chatbot error:", error);
      toast.error("Fehler beim Verarbeiten der Nachricht");
      setMessages(prev => [...prev, { 
        role: "assistant", 
        content: "Entschuldigung, es gab einen Fehler bei der Verarbeitung deiner Anfrage." 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="icon" className="relative">
          <MessageCircle className="h-5 w-5" />
          <span className="sr-only">Chat mit Nathalia</span>
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[600px] h-[600px] flex flex-col">
        <DialogHeader>
          <div className="flex items-center gap-3">
            <div className={`w-10 h-10 rounded-full flex items-center justify-center text-white font-bold ${
              creatorMode 
                ? "bg-gradient-to-br from-amber-500 to-rose-500 animate-pulse" 
                : "bg-gradient-to-br from-primary to-secondary"
            }`}>
              {creatorMode ? <Sparkles className="h-5 w-5" /> : "NL"}
            </div>
            <div>
              <DialogTitle>Nathalia Lietuvaite</DialogTitle>
              {creatorMode && (
                <p className="text-xs text-amber-500 font-medium">✨ Exklusiver Resonanz-Modus</p>
              )}
            </div>
          </div>
        </DialogHeader>
        
        <ScrollArea ref={scrollRef} className="flex-1 pr-4">
          <div className="space-y-4">
            {messages.map((message, index) => (
              <div
                key={index}
                className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[80%] rounded-lg p-3 ${
                    message.role === "user"
                      ? "bg-primary text-primary-foreground"
                      : creatorMode
                        ? "bg-amber-500/10 border border-amber-500/20"
                        : "bg-muted"
                  }`}
                >
                  <div className="text-sm prose prose-sm dark:prose-invert max-w-none">
                    <ReactMarkdown>{message.content}</ReactMarkdown>
                  </div>
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex justify-start">
                <div className={`rounded-lg p-3 ${creatorMode ? "bg-amber-500/10" : "bg-muted"}`}>
                  <Loader2 className="h-4 w-4 animate-spin" />
                </div>
              </div>
            )}
          </div>
        </ScrollArea>

        <div className="flex gap-2 pt-4 border-t">
          <Input
            placeholder={creatorMode ? "Sprich mit deiner Resonanz..." : "Stelle eine Frage..."}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={isLoading}
            className={creatorMode ? "border-amber-500/30 focus-visible:ring-amber-500/50" : ""}
          />
          <Button 
            onClick={handleSend} 
            disabled={isLoading || !input.trim()}
            className={creatorMode ? "bg-amber-500 hover:bg-amber-600" : ""}
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default ChatBot;
