set background=dark
set t_Co=256
set number
set nocompatible
filetype off
syntax on
color dracula

" Remap my leader
:let mapleader = ","

" Quick buffer resizing
if bufwinnr(1)
  map + <C-W>+
  map - <C-W>-
endif

set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()

" Terminal integration
Plugin 'lrvick/Conque-Shell'

" Vundle vim
Plugin 'VundleVim/Vundle.vim'

" Airline (status bar)
Plugin 'vim-airline/vim-airline'

" Airline themes
Plugin 'vim-airline/vim-airline-themes'

" NERD tree
Plugin 'scrooloose/nerdtree'

" Ctrl-p (file navigation)
Plugin 'kien/ctrlp.vim'

" Commenter
Plugin 'scrooloose/nerdcommenter'

" Dracula
Plugin 'dracula/vim'

" Jedi-vim (autocompletion for python)
Plugin 'davidhalter/jedi-vim'

" Supertab (autocompletion with tab)
Plugin 'ervandew/supertab'

" YouCompleteMe
" Plugin 'Valloric/YouCompleteMe'

" Python mode
Plugin 'python-mode/python-mode'

" Fugitive: a git wrapper
Plugin 'tpope/vim-fugitive'

" Syntax checking
Plugin 'scrooloose/syntastic'

" EasyMotion in VIM
Plugin 'easymotion/vim-easymotion'

" Python indentation
Plugin 'vim-scripts/indentpython.vim'

" Python PEP8 linter
Plugin 'nvie/vim-flake8'

" Python virtualenv support
Plugin 'jmcantrell/vim-virtualenv'

" Autoformatter
Plugin 'chiel92/vim-autoformat'

" Surround with brackets
Plugin 'tpope/vim-surround'

call vundle#end()

filetype plugin indent on
set laststatus=2
autocmd VimEnter * AirlineTheme dracula

" Syntastic settings
set statusline+=%#warningmsg#
set statusline+=%{SyntasticStatuslineFlag()}
set statusline+=%*

let g:syntastic_always_populate_loc_list = 1
let g:syntastic_auto_loc_list = 1
let g:syntastic_check_on_open = 1
let g:syntastic_check_on_wq = 0

" Navigations
nnoremap <C-J> <C-W><C-J>
nnoremap <C-K> <C-W><C-K>
nnoremap <C-L> <C-W><C-L>
nnoremap <C-H> <C-W><C-H>
map <F7> :tabp <CR>
map <F8> :tabn <CR>

" YCM python settings
" let g:ycm_path_to_python_interpreter = '/usr/bin/python'
" let g:ycm_python_binary_path = '/home/mirchev/workspace/learning_django/venv/bin/python'
" map <C-G> :YcmCompleter GoTo <CR>

"Autocomplete settings
noremap <F3> :Autoformat<CR>

" No line wrapping
set nowrap

" Supertab scroll down, not up
let g:SuperTabDefaultCompletionType = "<c-n>"

" jedi-vim settings
" Open new tab on go to definition
let g:jedi#use_tabs_not_buffers = 1

" python-mode settings
" turn off autocompletion, I'm using jedi-vim
let g:pymode_rope_completion = 0
" set python to python3
let g:pymode_python = 'python3'
" disable syntax checking, I'm using syntastic
let g:pymode_syntax = 0
" disable automatic python folding
let g:pymode_folding = 0
" enable linting on the fly
let g:pymode_lint_on_fly = 1
" set the max line length to 100
let g:pymode_options_max_line_length = 100
" hack to use ipdb instead of pdb
map <Leader>b Oimport ipdb; ipdb.set_trace() # BREAKPOINT<C-c>

" ConqueShell settings
let g:ConqueTerm_CloseOnEnd = 1
let g:ConqueTerm_InsertOnEnter = 1
let g:ConqueTerm_Color = 0
let g:ConqueTerm_PyVersion = 3
function MyConqueStartup(term)
	resize 10
endfunction

call conque_term#register_function('after_startup', 'MyConqueStartup')

" Rename refactor
function! Refactor()
	let word_to_replace = expand("<cword>")
	let replacement = input("new name:")
	execute('normal! \<C-G>[{V%')
	execute("'\<,'>s/" . word_to_replace . "/" . replacement . "/g")
endfunction

" Locally (local to block) rename a variable
nnoremap cr :call Refactor()<CR>
