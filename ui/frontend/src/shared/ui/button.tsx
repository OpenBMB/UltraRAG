import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/shared/lib/cn";

const buttonVariants = cva("ur-btn", {
  variants: {
    variant: {
      default: "ur-btn--default",
      ghost: "ur-btn--ghost",
      outline: "ur-btn--outline",
      danger: "ur-btn--danger",
    },
    size: {
      default: "ur-btn--md",
      sm: "ur-btn--sm",
      lg: "ur-btn--lg",
      icon: "ur-btn--icon",
    },
  },
  defaultVariants: {
    variant: "default",
    size: "default",
  },
});

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return <Comp className={cn(buttonVariants({ variant, size }), className)} ref={ref} {...props} />;
  },
);
Button.displayName = "Button";

export { Button };
