set background=dark
set t_Co=256
set number
set nocompatible
filetype off
syntax on
:set nowrap
set laststatus=2

" some neovim configuration
let g:python_host_prog = '/home/valor/.virtualenvs/neovim2/bin/python'
let g:python3_host_prog= '/home/valor/.virtualenvs/neovim3/bin/python'

" normal backspace
set backspace=2

" color of column lines, used to indicate maximal line width
hi ColorColumn ctermbg=8

" Source .vimrc on save
autocmd! bufwritepost .vimrc source %

" Unified clipboard (VIM and OS)
set pastetoggle=<F2>
set clipboard=unnamed

" remap the leader key
let mapleader = ','

" easier window navigation
nnoremap <C-J> <C-W><C-J>
nnoremap <C-K> <C-W><C-K>
nnoremap <C-L> <C-W><C-L>
nnoremap <C-H> <C-W><C-H>

" Sorting shortcut (in visual selection)
vnoremap <Leader>s :sort<CR>

" Shifting code blocks left and right (visual mode)
vnoremap < <gv " better indentation
vnoremap > >gv " better indentation

" Quick buffer resizing
if bufwinnr(1)
  map + <C-W>+
  map - <C-W>-
endif

" Plugins
filetype plugin on
filetype plugin indent on
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()

" Vundle vim (plugin manager)
Plugin 'VundleVim/Vundle.vim'

" Terminal goodies for neovim
Plugin 'vimlab/split-term.vim'

" Airline (status bar)
Plugin 'vim-airline/vim-airline'

" Airline themes (status bar)
Plugin 'vim-airline/vim-airline-themes'

" NERD tree (file tree)
Plugin 'scrooloose/nerdtree'

" Dracula (theme)
Plugin 'dracula/vim'

" Jedi-vim (autocompletion for python)
Plugin 'davidhalter/jedi-vim'

" Python-mode (refactoring and general support)
Plugin 'python-mode/python-mode'

" Python PEP8 linter
Plugin 'nvie/vim-flake8'

" Python virtualenv support
Plugin 'jmcantrell/vim-virtualenv'

" Supertab (autocompletion with tab)
Plugin 'ervandew/supertab'

" Python indentation
Plugin 'vim-scripts/indentpython.vim'

" Python docstrings
Plugin 'heavenshell/vim-pydocstring'

" YouCompleteMe (generic autocompletion)
Plugin 'Valloric/YouCompleteMe'

" Syntastic (generic linter)
Plugin 'vim-syntastic/syntastic'

" Fugitive: (a git wrapper)
Plugin 'tpope/vim-fugitive'

" EasyMotion in VIM (some improved motions)
Plugin 'easymotion/vim-easymotion'

" General autoformatting
Plugin 'Chiel92/vim-autoformat'

" CtrlP for quick file searches
Plugin 'kien/ctrlp.vim'

" Commenter (comment blocks of text)
Plugin 'scrooloose/nerdcommenter'

" Surround.vim (bracket management)
Plugin 'tpope/vim-surround'


call vundle#end()

" YCM settings
let g:ycm_autoclose_preview_window_after_completion = 1
nnoremap <C-G> :YcmCompleter GoTo<CR>

" disable python semantic completion, jedi-vim is better for me
let g:ycm_filetype_specific_completion_to_disable = {
	\ 'python' : 1, 
	\ 'gitcommit' : 1
	\}

" enable the dracula theme
color dracula

" Airline theme select
autocmd VimEnter * AirlineTheme dracula

" Syntastic settings
set statusline+=%#warningmsg#
set statusline+=%{SyntasticStatuslineFlag()}
set statusline+=%*

let g:syntastic_always_populate_loc_list = 1
let g:syntastic_auto_loc_list = 1
let g:syntastic_check_on_open = 0
let g:syntastic_check_on_wq = 0
let g:syntastic_enable_signs=0
let g:syntastic_echo_current_error=0
" syntastic is rather slow for python, use python-mode instead
let g:syntastic_mode_map = { 'passive_filetypes': ['python'] }

" Supertab scroll down, not up
let g:SuperTabDefaultCompletionType = "<c-n>"

" jedi-vim settings
" Open new tab on go to definition
let g:jedi#use_tabs_not_buffers = 1

" python-mode settings
" enable virtualenv
let g:pymode_virtualenv = 1
" turn off autocompletion, I'm using jedi-vim
let g:pymode_rope_completion = 0
" set python to python3
let g:pymode_python = 'python3'
" enable syntax checking, not using syntastic
let g:pymode_syntax = 1
" enable linting for python, not using syntastic
let g:pymode_lint = 1
" disable automatic python folding
let g:pymode_folding = 0
" disable linting on the fly
let g:pymode_lint_on_fly = 0
" set the max line length to 100
let g:pymode_options_max_line_length = 100
" hack to use ipdb instead of pdb
map <Leader>b Oimport ipdb; ipdb.set_trace() # BREAKPOINT<C-c>

" pydocstring
autocmd FileType python setlocal tabstop=4 shiftwidth=4 softtabstop=4 expandtab
nnoremap <C-S-d> :Pydocstring<CR>
