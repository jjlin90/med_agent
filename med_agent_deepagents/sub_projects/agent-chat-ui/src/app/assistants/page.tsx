"use client";

import { useState, FormEvent, Suspense } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { createClient } from "@/providers/client";
import { useQueryState } from "nuqs";
import { toast } from "sonner";
import { MedicalAgentLogoSVG } from "@/components/icons/medical-agent";
import { useRouter } from "next/navigation";

interface AssistantCreateForm {
  assistantId?: string;
}

function CreateAssistantContent() {
  const router = useRouter();
  const [apiUrl] = useQueryState("apiUrl");
  const [apiKey] = useState("");
  const [form, setForm] = useState<AssistantCreateForm>({
    assistantId: "",
  });
  const [isLoading, setIsLoading] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      const client = createClient(apiUrl || "", apiKey);
      // 使用默认值创建assistant，只需要assistantId
      await client.assistants.create({
        graphId: "agent", // 默认graphId
        name: form.assistantId || "default-assistant", // 使用assistantId作为name，或者默认名称
        description: "Default assistant created from UI", // 默认描述
        assistantId: form.assistantId,
        ifExists: "do_nothing", // 默认如果存在则不做任何操作
      });

      toast.success("Assistant created successfully!");
      if (form.assistantId) {
        router.push(`/?assistantId=${encodeURIComponent(form.assistantId)}`);
      } else {
        router.push("/");
      }
    } catch {
      toast.error("Failed to create assistant");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen flex-col bg-gray-50">
      <div className="border-b bg-white">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center gap-3">
            <MedicalAgentLogoSVG className="h-8 flex-shrink-0" />
            <h1 className="text-2xl font-semibold tracking-tight">
              Create Assistant
            </h1>
          </div>
        </div>
      </div>

      <div className="mx-auto w-full max-w-3xl flex-1 px-4 py-8 sm:px-6 lg:px-8">
        <div className="rounded-lg border bg-white shadow-sm">
          <div className="p-6">
            <h2 className="mb-4 text-lg font-medium">
              Assistant Configuration
            </h2>
            <form
              onSubmit={handleSubmit}
              className="grid gap-4"
            >
              <div className="grid gap-2">
                <Label htmlFor="assistantId">
                  Assistant ID <span className="text-red-500">*</span>
                </Label>
                <Input
                  id="assistantId"
                  name="assistantId"
                  value={form.assistantId}
                  onChange={handleChange}
                  placeholder="Enter assistant ID"
                  required
                />
              </div>
              <div className="flex justify-end">
                <Button
                  type="submit"
                  disabled={isLoading}
                >
                  {isLoading ? "Creating..." : "Create Assistant"}
                </Button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function CreateAssistantPage() {
  return (
    <Suspense
      fallback={
        <div className="flex min-h-screen items-center justify-center bg-gray-50">
          Loading...
        </div>
      }
    >
      <CreateAssistantContent />
    </Suspense>
  );
}
