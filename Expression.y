%{
	#include <iostream>
	#include <string>
    #include <cuda.h>
	#include <stdlib.h>
	#include <vector>
	#include <map>
	#include <pair>
	#include "proj_types.h"
	int yyerror (char* h);
	int yylex(void);
	map<std::string,int> column_map;  
%}
%union
{
    double value;
    std::string identifier;
	SelectQuery obj;
}
%start goal;
%type <SelectQuery> goal
%type <value> Integer
%type <> RepeatedParametersExpression
%type <> goal StatementStar Statement IfElseStatement MacroDefStatement MacroDefExpression MacroDefinition MacroDefinition_extended TypeDeclaration_extended TypeIdentifierExtended ClassDefinition PrimaryExpression Expression RepeatedParameters MethodDeclaration MethodDeclarationExtended MainClass TypeIdentifierComma
%type <std::string> Identifier  AccessSpecifier Type
%token Plus Minus Mult Div Modulo Semicolon Equal Comma OpeningBrace Identifier ClosingBrace OpeningBracket ClosingBracket OpeningSquareBracket ClosingSquareBracket Dot Return Main GreaterThan LessThan LessEqual GreaterEqual DoubleEqual NotEqual Or And Class While If Else True False New Length Define Hashtag Extends System Out Print Println Integer Public Private Protected Void Bool Not Static String This Int
%%
Select_Query: SelectCol DistinctQualifier Where Exp GroupExp OrderExp Limit Value
{

}
| SelectCol Where Exp OrderExp Limit Value
{

}
| SelectCol Where Exp OrderExp
{

}
| SelectCol Where Exp
{

}
| SelectCol Where 
;
SelectCol: Identifier MultiCol
{
	$$ = $2;
	$$.push_back($1);
}
| Multiply
{
	$$ = *(new vector<std::string>);
	for(auto it: column_map)
		$$.push_back(it.first);
};
MultiCol: MultiCol Comma Identifier
{
	$$ = $1;
	$1.push_back($3);
}	
| 
{
	$$ = *(new vector<std::string>)
};
GroupExp: Group By Exp
{
	$$ = $3;
}
| 
{
	$$ = NULL;
};
OrderExp: Order By Exp Order ExpList
{
	$$ = $5;
	$$.push_back(std::make_pair($3,$4));
}
| Order By Exp ExpList
{
	$$ = $4;
	$$.push_back(std::make_pair($3,true));
}
| 
{
	$$ = NULL;
}
ExpList: ExpList Comma Exp
{
	$$ = $1;
	$$.push_back(std::make_pair($2,true));
}
| ExpList Comma Exp Order 
{
	$$ = $1;
	$$.push_back(std::make_pair($2,$3));
}
| 
{
	$$ = *(new vector<std::pair<ExpressionNode,bool>>);
};
Exp: Exp Or Exp1
{
	$$ = new ExpressionNode("or");
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
}
| Exp And Exp1
{
	$$ = new ExpressionNode("and");
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
}
| Exp1
{
	$$ = $1;
};

Exp1: Exp1 Greater Exp2
{
	$$ = new ExpressionNode("greater");
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
}
| Exp1 Lesser Exp2
{
	$$ = new ExpressionNode("greater");
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
}
| Exp1 GreaterEqual Exp2
{
	$$ = new ExpressionNode("GreaterEqual");
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
}
| Exp1 LesserEqual Exp2
{
	$$ = new ExpressionNode("LesserEqual");
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
}
| Exp1 Equal Exp2
{
	$$ = new ExpressionNode("Equal");
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
}
| Exp1 NotEqual Exp2
{
	$$ = new ExpressionNode("NotEqual");
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
};
| Exp2
{
	$$ = $1;
}

Exp2: Exp2 Plus Exp3
{
	$$ = new ExpressionNode("Plus");
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
}
| Exp2 Minus Exp3
{
	$$ = new ExpressionNode("NotEqual");
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
}
| Exp3
{
	$$ = $1;
};
Exp3: Expr3 Multiply Term
{
	$$ = new ExpressionNode("Multiply");
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
}
| Expr3 Divide Term
{
	$$ = new ExpressionNode("Divide");
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
}
| Expr3 Modulo Term
{
	$$ = new ExpressionNode("NotEqual");
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
};
Term: Column
{

}
| Value
{
	$$ = new ExpressionNode;
	$$->left_hand_term = NULL;
	$$->right_hand_term = NULL;
	$$->
}
| OpeningBracket Exp ClosingBracket
{
	$$ = $2;
}
;
%%
int yyerror(char *s)
{
	printf ("// Failed to parse macrojava code.");
	return 0;
  
}
int main ()
{
	macro_table = (table*)(malloc(sizeof(table)));
	yyparse();
	return 0;
}
