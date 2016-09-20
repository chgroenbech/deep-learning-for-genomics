\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{report}[2016/05/14 v0.1 Report class]

\usepackage{xifthen}					% Implements if clauses

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{memoir}}

\ProcessOptions\relax

\LoadClass[a4paper,11pt,oneside,twocolumn,openany,article]{memoir}

% Load standard packages
\usepackage{fixltx2e}					% Fixes to LaTeX2e
\usepackage[utf8]{inputenc}				% Input encoding: UTF-8
\usepackage[T1]{fontenc}				% Font encoding: T1
\usepackage{graphicx}					% Graphics package

% Load localization packages
\usepackage{babel}						% For languages in document class
\usepackage{datetime}					% Include time commands
\usepackage[cleanlook]{isodate} 		% Date formatting: D MMMM YYYY

% Load typeface package
\usepackage[light, easyscsl, noDcommand]{kpfonts}
% Typeface: The Kepler project
% Also loads AMS packages

% Load math packages
\usepackage{amsmath}
\usepackage{mathtools}					% Extra math commands
\usepackage{bm}							% Bold math
\usepackage{slashed}
\usepackage{math}

% Load packages for scientific articles
\usepackage{siunitx}					% SI units
% \usepackage[square, numbers]{natbib}	% Natural science references

% Load miscellaneous packgaes
\usepackage{xcolor}						% Extended package for colouring
\usepackage{ellipsis}					% Fixes ellipses before letters
\usepackage{xspace}						% Add space after text macros with \xspace
\usepackage{paralist}					% Lists in paragraphs
\usepackage{afterpage}
\usepackage{flafter}
\usepackage{grffile}
\usepackage{keyval}
\usepackage{balance}                    % Balance text columns
% \usepackage{flushend}
\usepackage{tikz}

% Load cross reference packages
\usepackage{hyperref}					% PDF links for cross-references, citations, etc.
\usepackage{bookmark}					% Fixes aspects of the hyperref package

% Note: For the hyperref package to work properly it should be loaded after all other packages except the bookmark package, but before \newcommand's and such.

% Colours
\definecolor{dark-red}{rgb}{0.4,0.15,0.15}
\definecolor{dark-blue}{rgb}{0.15,0.15,0.4}
\definecolor{medium-blue}{rgb}{0,0,0.5}
\definecolor{light-grey}{rgb}{0.98,0.98,0.98}

% Page setup
\settrimmedsize{297mm}{210mm}{*}
\setlength{\trimtop}{0pt}
\setlength{\trimedge}{\stockwidth}
\addtolength{\trimedge}{-\paperwidth}
% \settypeblocksize{*}{400pt}{1.618}
% \setulmargins{4cm}{*}{*}
\settypeblocksize{241mm}{178mm}{*}
\setulmargins{25mm}{*}{*}
\setlrmargins{*}{*}{1}
\setmarginnotes{0pt}{0pt}{\onelineskip}
\setheadfoot{\onelineskip}{2\onelineskip}
\setheaderspaces{*}{2\onelineskip}{*}
\setcolsepandrule{2em}{0pt}
\checkandfixthelayout
\flushbottom

% Title
\setlength{\droptitle}{-\onelineskip}
\newcommand{\course}[1]{\gdef\thecourse{#1}}
\newcommand{\thecourse}{}
\newcommand{\affiliation}[1]{\gdef\theaffiliation{#1}}
\newcommand{\theaffiliation}{}
\renewcommand{\maketitlehooka}{%
	\begin{center}
		\normalsize \MakeUppercase
		\thecourse
	\end{center}
	\vspace{-1.5\onelineskip}
}
\pretitle{\begin{center} \Large \bfseries \MakeUppercase}
\posttitle{\end{center}}
\preauthor{\begin{center}
			\large \itshape
			\begin{tabular}[t]{@{}l}}
\postauthor{\end{tabular}\end{center}}
\renewcommand*\and{%
	\end{tabular}%
	\hskip 1em \relax
	\begin{tabular}[t]{l}}
\renewcommand*{\andnext}{%
	\end{tabular}\\ \begin{tabular}[t]{@{}l}}
\renewcommand{\maketitlehookc}{
	\begin{center}
		\large
		\theaffiliation
	\end{center}
}
\predate{\begin{center} \footnotesize}
\postdate{\end{center}}
\renewcommand{\maketitlehookd}{
	\aftergroup\@afterindentfalse
	\aftergroup\@afterheading
}

% Page style
\makepagestyle{nice}
% \makeevenhead {nice}{}{}{}
\makeoddhead  {nice}{\theauthor}{}{\thecourse}
\makeheadrule {nice}{\textwidth}{\normalrulethickness}
% \makeevenfoot {nice}{\}{}{}
\makeoddfoot  {nice}{}{\thepage}{}
\makepsmarks  {nice}{%
	\clearmark{chapter}
	\clearmark{section}
	\clearmark{subsection}
	\clearmark{subsubsection}
	\clearplainmark{toc}
	\clearplainmark{lot}
	\clearplainmark{lof}
	\clearplainmark{bib}
	\clearplainmark{index}
}
\nouppercaseheads % No upper cases.

\copypagestyle{plain}{nice}
\makeevenhead {plain}{}{}{}
\makeoddhead  {plain}{}{}{}
\makeheadrule {plain}{\textwidth}{0pt}
\makeevenfoot {plain}{}{\thepage}{}
\makeoddfoot  {plain}{}{\thepage}{}

\mergepagefloatstyle{nicefloats}{nice}{plain}

\pagestyle{nicefloats}

% Document division
\setsecnumdepth{subparagraph}
\maxsecnumdepth{subparagraph}
\counterwithout{section}{chapter}
\renewcommand*{\chaptitlefont}{\centering\normalfont\large\bfseries}
\renewcommand*{\printchaptertitle}[1]{\chaptitlefont\MakeUppercase{#1}}
\setsecheadstyle{\centering\large\bfseries\MakeUppercase}
\setsubsecheadstyle{\flushleft\bfseries}
\setsubsubsecheadstyle{\flushleft\itshape}
\setsechook{\setsecnumformat{\csname the##1\endcsname.\enspace}}
\addtodef{\appendix}{}{\def\thesection{\alph{section}}\nobalance}

% ToC
\settocdepth{subparagraph}
\maxtocdepth{subparagraph}

% Floats and other input
\captionnamefont{\bfseries}
\captiondelim{\ \ }
\captiontitlefinal{}
\newsubfloat{figure}
\tightsubcaptions

% Footnotes
\setlength{\footmarkwidth}{0em}
\setlength{\footmarksep}{-\footmarkwidth}
% \setlength{\footparindent}{0em}

% Units
\sisetup{%
	per=slash,
	round-precision = 5,
	round-mode = off,
	range-units = single,
	range-phrase = \text{--},
	list-separator = {\text{, }},
	list-final-separator = {\text{, and }}
}
\DeclareSIUnit{\deg}{deg}
\DeclareSIUnit{\year}{yr}

% Revisioning
\newcommand{\change}[1]{\textcolor{red}{\emph{(#1)}}}
\newcommand{\rephrase}[1]{\textcolor{orange}{#1}}

% References
\bibliographystyle{ieeetr}
\addto\captionsbritish{
	\renewcommand\bibname{References}
	\renewcommand{\bibsection}{
		\section{\bibname}
		\prebibhook
	}
}
\renewcommand{\postbibhook}{\balance}

% Text formatting
\newcommand{\newterm}[1]{\textbf{#1}}
\newcommand{\programminglanguage}[1]{\textsf{#1}}
\newcommand{\library}[1]{\textsf{#1}}
\newcommand{\function}[1]{\texttt{#1}}
\newcommand{\class}[1]{\textbf{\texttt{#1}}}
\newcommand{\file}[1]{\texttt{#1}}
\newcommand{\variable}[1]{\textit{\texttt{#1}}}
\newcommand{\code}[1]{\texttt{#1}}

% Cross references
\hypersetup{
	colorlinks,
	linkcolor={dark-red},
	citecolor={dark-blue},
	urlcolor={medium-blue},
	% bookmarks=true,
	bookmarksnumbered=true,
	unicode=true,
	plainpages=false,
	% pdfborder={0 0 0},
}
\bookmarksetup{startatroot}
\AtBeginDocument{%
	\hypersetup{pdftitle={\thetitle}, pdfauthor={\theauthor}, pdfsubject={}}
}
