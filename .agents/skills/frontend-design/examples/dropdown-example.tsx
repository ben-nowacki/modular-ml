// Radix DropdownMenu styled with Tailwind + project tokens.
// Pattern reference — adapt token names / imports to the project manifest.
import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import { MoreHorizontal } from "lucide-react";
import { cn } from "@/lib/utils";

const contentCls = cn(
  "z-50 min-w-[10rem] rounded-md border border-border bg-surface p-1 shadow-md",
  "data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95"
);
const itemCls = cn(
  "flex cursor-default select-none items-center gap-2 rounded-sm px-2 py-1.5 text-xs text-foreground",
  "outline-none data-[highlighted]:bg-muted",
  "data-[disabled]:pointer-events-none data-[disabled]:opacity-50"
);

export function RowActions({
  onEdit,
  onDelete,
}: {
  onEdit: () => void;
  onDelete: () => void;
}) {
  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger
        aria-label="Row actions"
        className="rounded p-1 text-muted-foreground hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        <MoreHorizontal className="size-4" />
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content align="end" sideOffset={4} className={contentCls}>
          <DropdownMenu.Item className={itemCls} onSelect={onEdit}>
            Edit
          </DropdownMenu.Item>
          <DropdownMenu.Separator className="my-1 h-px bg-border" />
          <DropdownMenu.Item
            className={cn(itemCls, "text-destructive data-[highlighted]:bg-destructive/10")}
            onSelect={onDelete}
          >
            Delete
          </DropdownMenu.Item>
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}
