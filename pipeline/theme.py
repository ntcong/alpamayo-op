"""CSS shared by the report pages: dark fallback values for the theme variables.

:where() gives the :root rule zero specificity, so a host page that defines the same variables wins.
"""

ROOT_CSS = (
    ":where(:root){--background:#0d0d0d;--foreground:#f2f2f2;--border:#333;--muted-foreground:#999;"
    "--chart-1:#3b82f6;--chart-2:#2dd4bf;--chart-3:#f5b301;--chart-4:#a78bfa;--chart-5:#f472b6;"
    "--success:#22c55e;--radius:8px;"
    "--font-sans:system-ui,-apple-system,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;"
    "--font-mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}\n"
    "body{background:var(--background);color:var(--foreground);padding:12px}\n"
)
