%{
#include <iostream>
#include <string.>
#include <stdlib.h>
#include "P1.tab.h"
%}

/*Flex definitions*/
value ^(-?)(0|([1-9][0-9]*))(\\.[0-9]+)?$ 
identifier [a-zA-Z_][0-9a-zA-Z_]*
space "\t"|"\n"|"\r"|"\f"|" "
%% 
{value}  {yylval.value = stoi(yytext);return Value;}
"+" {return Plus;}
"-"  {return Minus;}
"*" {return Mult;}
"/" {return Div;}
"%" {return Modulo;}
"==" {return Equal;}
"!=" {return NotEqual;}
">=" {return GreaterEqual;}
"<=" {return LesserEqual;}
">" {return Greater;}
"<" {return Lesser;}
"!" {return Not;}
"(" {return OpeningBracket;}
")" {return ClosingBracket;}
"," {return Comma;}
"&&" {return And;}
"||" {return Or;}
"LIMIT" {return Limit;}
"ORDER" {return Order;}
"BY" {return By;}
"GROUP" {return Group;}
"WHERE" {return Where;}
"DISTINCT" {return Distinct;}
"ASC" {return Ascending;}
"DESC" {return Descending;}
"MIN" {return Minimum;}
"MAX" {return Maximum;}
"AVG" {return Average;}
"SUM" {return Sum;}
"VAR" {return Variance;}
"STDDEV" {return StandardDeviaton;}
"COUNT" {return Count;}
{identifier} {yylval.identifier = *(new std::string(yytext)); return Identifier;}
{space} {;}
%%
