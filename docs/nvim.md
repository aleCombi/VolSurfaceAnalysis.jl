# Exploring this repo in Neovim

A VS Code translation, for when the sidebar reflex kicks in. Leader is
SPACE. The Julia language server (`julials`) attaches on any `.jl`
buffer under this project, so the "go to" family works on real symbols
rather than plain text.

## Open a file

| | |
|---|---|
| `<Space>ff` | find files by name (VS Code: Ctrl-P) |
| `<Space>fg` | live grep the repo (Ctrl-Shift-F) |
| `<Space>fb` | switch buffers -- this is the tab bar |
| `<Space>fd` | list diagnostics |

Inside a telescope window: type to filter, `Ctrl-n` / `Ctrl-p` to move,
`Enter` to open, `Ctrl-v` / `Ctrl-x` to open in a split, `Ctrl-u` /
`Ctrl-d` to scroll the preview, `Esc Esc` to cancel.

## Follow the code

| | |
|---|---|
| `gd` | go to definition (F12) |
| `gr` | list references (Shift-F12) |
| `K` | hover: docstring and signature |
| `Ctrl-o` | go back |
| `Ctrl-i` | go forward |
| `<Space>rn` | rename symbol |
| `<Space>ca` | code action |

`Ctrl-o` / `Ctrl-i` walk a jumplist, so you can follow `gd` several
definitions deep across files and unwind without tracking where you
were.

## Move inside one file

`gg` / `G` top and bottom. `{` / `}` jump by blank-line block. `%` to
the matching bracket. `*` searches the word under the cursor, `n` / `N`
step through matches, `Esc` clears the highlight. `:42` goes to line
42. `ma` sets mark `a` and `'a` returns to it. `Ctrl-u` / `Ctrl-d`
scroll half a page.

## Folder structure

Neovim's built-in file browser, netrw, needs no plugin:

| | |
|---|---|
| `:Lexplore` | toggle a tree sidebar (VS Code: the explorer) |
| `:Explore` | browse in the current window |
| `:Explore %:h` | browse the directory of the current file |

In the listing: `Enter` opens a file or expands a directory, `-` goes
up a level, `d` makes a directory, `%` makes a file, `D` deletes. For a
tree rather than a flat listing, `:let g:netrw_liststyle=3` (put it in
`init.lua` as `vim.g.netrw_liststyle = 3` to make it the default).

Outside the editor, `git ls-files | awk -F/ '{print $1}' | uniq -c`
gives a quick census of the top level.

## Splits

`:vsplit` and `:split` divide the window; `:q` closes one and `:bd`
closes the buffer. `Ctrl-h/j/k/l` moves between splits and keeps going
past the edge of Neovim into the tmux panes -- vim-tmux-navigator makes
the cockpit one grid.

## Git

gitsigns marks changed lines in the sign column. `<Space>lg` opens
lazygit in a terminal buffer; `Esc Esc` leaves terminal mode, `:bd`
closes it.

## Wired to the rest of the workspace

| | |
|---|---|
| `<Space>js` | send the line, or the visual selection, to the Julia REPL pane |
| `<Space>jt` | run the test suite in the shell window |
| `:Where` | print `file:line`, to tell an agent what you are looking at |

The editor also runs an RPC server on a socket derived from its cwd,
which is how `ws open` puts a file on screen from outside. See the
`ws` script and `~/.config/nvim/init.lua`.

## The habit to build

The sidebar is there if you want it, but reaching for it every time is
the slow path. Faster: `<Space>ff` when you know the name, `<Space>fg`
when you know a string inside it, `gd` when the symbol is already on
screen, and `Ctrl-o` to come back. Keep netrw for the times you want to
see the shape of a directory you do not know yet.
