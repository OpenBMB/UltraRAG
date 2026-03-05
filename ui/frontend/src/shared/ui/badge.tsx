import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/shared/lib/cn";

const badgeVariants = cva("ur-badge", {
  variants: {
    variant: {
      default: "ur-badge--default",
      secondary: "ur-badge--secondary",
      success: "ur-badge--success",
      danger: "ur-badge--danger",
    },
  },
  defaultVariants: {
    variant: "default",
  },
});

type BadgeProps = React.HTMLAttributes<HTMLDivElement> & VariantProps<typeof badgeVariants>;

export function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
}
