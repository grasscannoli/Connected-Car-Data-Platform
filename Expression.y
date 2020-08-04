%{
	#include <iostream>
	#include <string>
	#include <stdlib.h>
	#include <vector>
	#include <map>
	#include <pair>
	#include <cmath>
	#include "proj_types.h"
	void yyerror (SelectQuery* select_query,const char* error_msg);
	int yyparse(SelectQuery* select_query);
	int yylex_destroy(void);
	int yy_scan_string(const char*);
	map<std::string,int> column_map;  
%}
%union
{
    double value;
    std::string identifier;
	SelectQuery* SelectObject;
	bool distinct;
	ExpressionNode* expression;
	std::vector<std::string> name_list;
	std::vector<std::pair<std::string,ExpressionNode*>> expression_list;
	std::vector<std::pair<ExpressionNode*,bool>> order_list;
}
%parse-param {SelectQuery* select_query}
%start goal;
%type <SelectObject> goal Select_Query
%type <value> LimitExp
%type <distinct> DistinctQualifier
%type <expression> WhereCondition Column GroupExp
%type <expression_list> MultiAggCol AggCol
%type <name_list> SelectCol MultiCol
%type <order_list> OrderExp ExpList
%type <expression> Exp1 Exp2 Exp3 Exp 
%type <identifier> Column AggregateFunction 
%token Plus Minus Mult Div Modulo NotEqual Equal Greater GreaterEqual Lesser LesserEqual Or And Not Where Order Group By Limit Distinct Ascending Descecding Comma OpeningBracket ClosingBracket Maximum Minimum Average Variance StandardDeviation Count Sum
%%
goal: Select_Query
{
	$$ = $1;
	*select_query = $$;
};
Select_Query: SelectCol DistinctQualifier WhereCondition GroupExp OrderExp LimitExp
{
	$$ = new SelectQuery();
	$$->select_coumns = $1;
	$$->distinct_query = $2;
	$$->select_expression = $3;
	$$->group_term = $4;
	$$->order_term = $5;
	$$->limit_term = $6;
}
| AggCol DistinctQualifier WhereCondition GroupExp OrderExp LimitExp
{
	$$ = new SelectQuery();
	$$->aggregate_coumns = $1;
	$$->distinct_query = $2;
	$$->select_expression = $3;
	$$->group_term = $4;
	$$->order_term = $5;
	$$->limit_term = $6;
};
DistinctQualifier: Distinct
{
	$$ = true;
}
|
{
	$$ = false;
};
WhereCondition: Where Exp
{
	$$ = $2;
}
| 
{
	$$ = NULL;
};
LimitExp: Limit Value
{
	$$ = $2;
};
| 
{
	$$ = -1;
}
AggCol: AggregateFunction OpeningBracket Exp ClosingBracket MultiAggCol
{
	$$ = $5;
	$$.push_back(std::make_pair($1,$3));
}
|  
{
	$$ =  *(new std::vector<std::pair<std::string,ExpressionNode*>>);
};
MultiAggCol: MultiAggCol Comma AggregateFunction OpeningBracket Exp ClosingBracket
{
	$$ = $1;
	$$.push_back(std::make_pair($3,$5));
}
| 
{
	$$ = (new std::vector<std::pair<std::string,ExpressionNode*>>);
};
SelectCol: Identifier MultiCol
{
	$$ = $2;
	$$.push_back($1);
}
| Mult
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
	$$ = *(new vector<std::string>);
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
};
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
	$$->type = 1;
}
| Exp And Exp1
{
	$$ = new ExpressionNode("and");
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	$$->type = 1;
}
| Not Exp1
{
	$$ = new ExpressionNode();
	$$->exp_operator = "Not";
	$$->left_hand_term = $1;
	if($$->type != 1)
		YYABORT;
	$$->type = 1;
};

Exp1: Exp1 Greater Exp2
{
	$$ = new ExpressionNode("greater");
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	if($1->type ==  1 || $3->type == 1)
		YYABORT;
	$$->type = 1;
}
| Exp1 Lesser Exp2
{
	$$ = new ExpressionNode("greater");
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	if($1->type ==  1 || $3->type == 1)
		YYABORT;
	$$->type = 1;
}
| Exp1 GreaterEqual Exp2
{
	$$ = new ExpressionNode("GreaterEqual");
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	if($1->type ==  1 || $3->type == 1)
		YYABORT;
	$$->type = 1;
}
| Exp1 LesserEqual Exp2
{
	$$ = new ExpressionNode("LesserEqual");
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	if($1->type ==  1 || $3->type == 1)
		YYABORT;
	$$->type = 1;
}
| Exp1 Equal Exp2
{
	$$ = new ExpressionNode("Equal");
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	if($1->type ==  1 || $3->type == 1)
		YYABORT;
	$$->type = 1;
}
| Exp1 NotEqual Exp2
{
	$$ = new ExpressionNode("NotEqual");
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	if($1->type ==  1 || $3->type == 1)
		YYABORT;
	$$->type = 1;
}
| Exp2
{
	$$ = $1;
};
Exp2: Exp2 Plus Exp3
{
	$$ = new ExpressionNode("Plus");
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	if($1->type ==  1 || $3->type == 1)
		YYABORT;
	$$->type = max($1->type,$2->type);
}
| Exp2 Minus Exp3
{
	$$ = new ExpressionNode("NotEqual");
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	if($1->type ==  1 || $3->type == 1)
		YYABORT;
	$$->type = max($1->type,$3->type);
}
| Exp3
{
	$$ = $1;
};
Exp3: Exp3 Multiply Term
{
	$$ = new ExpressionNode("Multiply");
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	if($1->type ==  1 || $3->type == 1)
		YYABORT;
	$$->type = 3;
}
| Exp3 Divide Term
{
	$$ = new ExpressionNode("Divide");
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	if($1->type ==  1 || $3->type == 1)
		YYABORT;
	$$->type = 3;
}
| Exp3 Modulo Term
{
	$$ = new ExpressionNode("NotEqual");
	$$->left_hand_term = $1;
	$$->right_hand_term = $3;
	if($1->type !=  2 || $3->type != 2)
		YYABORT;
	$$->type = 2;
};
Term: Column
{
	$$ = new ExpressionNode();
	$$->column_name = $1;
	$$->type = column_map[$1];
}
| Value
{
	$$ = new ExpressionNode();
	$$->value = $1;
	$$->type = (floor($1) == $1)?2:3;
}
| OpeningBracket Exp ClosingBracket
{
	$$ = $2;
};
%%
int yyerror(SelectQuery* select_query,const char* error_msg)
{
	YYABORT;
}
SelectQuery* process_query(std::string query)
{
	if(column_map.size() == 0)
	{
		column_map["vehicle_id"] = 2;
		column_map["database_index"] = 2;
		column_map["oil_life_pct"] = 3;
		column_map["tire_p_fl"] = column_map["tire_p_fr"] = column_map["tire_p_rl"] = column_map["tire_p_rr"] = 3;
		column_map["batt_volt"] = 3;
		column_map["fuel_percentage"] = 3;
		column_map["accel"] = 1;
		column_map["seatbelt"] = column_map["door_lock"] = column_map["hard_brake"] = column_map["gear_toggle"] = 1;
		column_map["clutch"] = column_map["hard_steer"] = 1;  
		column_map["speed"] = column_map["distance"] = 3;
	}
	SelectQuery* select_query;
	yy_scan_string(query.c_str());
	yyparse(select_query);
	yylex_destroy();
	return select_query;
}

