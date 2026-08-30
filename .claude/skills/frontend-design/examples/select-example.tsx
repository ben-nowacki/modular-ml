// Radix Select styled with Tailwind + project tokens.
// Pattern reference — adapt token names / imports to the project manifest.
import * as Select from "@radix-ui/react-select";
import { Check, ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

export function StatusSelect({
  value,
  onValueChange,
  options,
}: {
  value: string;
  onValueChange: (v: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <Select.Root value={value} onValueChange={onValueChange}>
      <Select.Trigger
        aria-label="Status"
        className={cn(
          "inline-flex h-8 items-center justify-between gap-2 rounded-md border border-border",
          "bg-surface px-3 text-xs text-foreground",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
          "data-[placeholder]:text-muted-foreground"
        )}
      >
        <Select.Value placeholder="Select…" />
        <Select.Icon>
          <ChevronDown className="size-4 text-muted-foreground" />
        </Select.Icon>
      </Select.Trigger>
      <Select.Portal>
        <Select.Content
          position="popper"
          sideOffset={4}
          className="z-50 overflow-hidden rounded-md border border-border bg-surface shadow-md"
        >
          <Select.Viewport className="p-1">
            {options.map((o) => (
              <Select.Item
                key={o.value}
                value={o.value}
                className={cn(
                  "relative flex cursor-default select-none items-center rounded-sm py-1.5 pl-7 pr-2 text-xs text-foreground",
                  "outline-none data-[highlighted]:bg-muted data-[state=checked]:font-medium"
                )}
              >
                <Select.ItemIndicator className="absolute left-2 inline-flex items-center">
                  <Check className="size-3.5" />
                </Select.ItemIndicator>
                <Select.ItemText>{o.label}</Select.ItemText>
              </Select.Item>
            ))}
          </Select.Viewport>
        </Select.Content>
      </Select.Portal>
    </Select.Root>
  );
}
