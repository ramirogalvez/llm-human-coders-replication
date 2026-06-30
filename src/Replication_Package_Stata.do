**** Regression and t-tests GPT **** 

/*******************************************************************************

* Created: Andrés Gago
* Date: 02/10/2024

*******************************************************************************/

clear all
set more off

local output_dir "../data/output_data"
local tables_dir "`output_dir'/tables"
capture mkdir "../data/output_data"
capture mkdir "`tables_dir'"

capture confirm file "`output_dir'/data_for_regs.txt"
if _rc {
    display as error "Missing `output_dir'/data_for_regs.txt. Run src/full_analysis.py first, or restore the provided output data."
    exit 601
}

capture which outreg2
if _rc {
    display as error "Missing Stata package outreg2. Install it with: ssc install outreg2"
    exit 199
}

capture which esttab
if _rc {
    display as error "Missing Stata package estout/esttab. Install it with: ssc install estout"
    exit 199
}

import delimited using "`output_dir'/data_for_regs.txt", clear varnames(1) case(lower)

//Generate dummies for each method

gen qualtrics=1 if coder=="Q"
replace qualtrics=0 if coder!="Q"
label var qualtrics "Qualtrics"

gen gpt35=1 if coder=="C35"
replace gpt35=0 if coder!="C35"
label var gpt35 "GPT-3.5"

gen gpt4=1 if coder=="C4"
replace gpt4=0 if coder!="C4"
label var gpt4 "GPT-4"

gen claude=1 if coder=="CL"
replace claude=0 if coder!="CL"
label var claude "Claude"

gen sonet=1 if coder=="CS"
replace sonet=0 if coder!="CS"
label var sonet "Sonet"

//Label variables
label var score_q1 "T1 (FScore)"
label var score_q2 "T2 (MAE)"
label var score_q3 "T3 (Acu)"
label var score_q4 "T4 (Acu)"
label var score_q5 "T5 (Acu)"
label var score_all_perfect "All (Prop)"
label var long_article "Long"


//Dummify variables

replace difficult_q1="1" if difficult_q1=="True"
replace difficult_q1="0" if difficult_q1=="False"

replace difficult_q2="1" if difficult_q2=="True"
replace difficult_q2="0" if difficult_q2=="False"

replace difficult_q3="1" if difficult_q3=="True"
replace difficult_q3="0" if difficult_q3=="False"

replace difficult_q4="1" if difficult_q4=="True"
replace difficult_q4="0" if difficult_q4=="False"

replace difficult_q5="1" if difficult_q5=="True"
replace difficult_q5="0" if difficult_q5=="False"

replace difficult_any="1" if difficult_any=="True"
replace difficult_any="0" if difficult_any=="False"

destring difficult_q1, replace
destring difficult_q2, replace
destring difficult_q3, replace
destring difficult_q4, replace
destring difficult_q5, replace
destring difficult_any, replace

egen participant_id=group(responseid)

// Regression Table C1

* We compare each method against human coders (the omitted category), controlling for long and difficult articles. We also compare the oldest and newest model of Openia, the oldest and newest model of Anthropic, and the newest models of Openia and Anthorpic. 

gen difficult=difficult_q1
label var difficult "Difficult"

reg score_q1 gpt35 gpt4 claude sonet difficult long_article, cluster(newsid)
test gpt35=gpt4
local p1 `r(p)'
local p1=round(`p1',0.001)
test gpt35=claude
local p2 `r(p)'
local p2=round(`p2',0.001)
test gpt4=sonet
local p3 `r(p)'
local p3=round(`p3',0.001)
outreg2 using "`tables_dir'/Tab_C1_main_reg.tex", replace label symbol(***, **, *) tex(frag) ///
addtext(p-value: $\beta_{GPT3.5}=\beta_{GPT4}$, `p1', p-value: $\beta_{GPT3.5}=\beta_{Claude}$, `p2', p-value: $\beta_GPT4=\beta_{Sonet}$, `p3') ///
cons nonotes nor2
	
	

replace difficult=difficult_q2
	
reg score_q2 gpt35 gpt4 claude sonet difficult long_article, cluster(newsid)
test gpt35=gpt4
local p1 `r(p)'
local p1=round(`p1',0.001)
test gpt35=claude
local p2 `r(p)'
local p2=round(`p2',0.001)
test gpt4=sonet
local p3 `r(p)'
local p3=round(`p3',0.001)
outreg2 using "`tables_dir'/Tab_C1_main_reg.tex", append label symbol(***, **, *) tex(frag) ///
addtext(p-value: $\beta_{GPT3.5}=\beta_{GPT4}$, `p1', p-value: $\beta_{GPT3.5}=\beta_{Claude}$, `p2', p-value: $\beta_GPT4=\beta_{Sonet}$, `p3') ///
cons nonotes nor2
	
	
replace difficult=difficult_q3
	
reg score_q3 gpt35 gpt4 claude sonet difficult long_article, cluster(newsid)
test gpt35=gpt4
local p1 `r(p)'
local p1=round(`p1',0.001)
test gpt35=claude
local p2 `r(p)'
local p2=round(`p2',0.001)
test gpt4=sonet
local p3 `r(p)'
local p3=round(`p3',0.001)
outreg2 using "`tables_dir'/Tab_C1_main_reg.tex", append label symbol(***, **, *) tex(frag) ///
addtext(p-value: $\beta_{GPT3.5}=\beta_{GPT4}$, `p1', p-value: $\beta_{GPT3.5}=\beta_{Claude}$, `p2', p-value: $\beta_GPT4=\beta_{Sonet}$, `p3') ///
cons nonotes nor2
	
replace difficult=difficult_q4

reg score_q4 gpt35 gpt4 claude sonet difficult long_article, cluster(newsid)
test gpt35=gpt4
local p1 `r(p)'
local p1=round(`p1',0.001)
test gpt35=claude
local p2 `r(p)'
local p2=round(`p2',0.001)
test gpt4=sonet
local p3 `r(p)'
local p3=round(`p3',0.001)
outreg2 using "`tables_dir'/Tab_C1_main_reg.tex", append label symbol(***, **, *) tex(frag) ///
addtext(p-value: $\beta_{GPT3.5}=\beta_{GPT4}$, `p1', p-value: $\beta_{GPT3.5}=\beta_{Claude}$, `p2', p-value: $\beta_GPT4=\beta_{Sonet}$, `p3') ///
cons nonotes nor2
	
	

replace difficult=difficult_q5
	
reg score_q5 gpt35 gpt4 claude sonet difficult long_article, cluster(newsid)
test gpt35=gpt4
local p1 `r(p)'
local p1=round(`p1',0.001)
test gpt35=claude
local p2 `r(p)'
local p2=round(`p2',0.001)
test gpt4=sonet
local p3 `r(p)'
local p3=round(`p3',0.001)
outreg2 using "`tables_dir'/Tab_C1_main_reg.tex", append label symbol(***, **, *) tex(frag) ///
addtext(p-value: $\beta_{GPT3.5}=\beta_{GPT4}$, `p1', p-value: $\beta_{GPT3.5}=\beta_{Claude}$, `p2', p-value: $\beta_GPT4=\beta_{Sonet}$, `p3') ///
cons nonotes nor2
	
	
replace difficult=difficult_any
	
reg score_all_perfect gpt35 gpt4 claude sonet difficult long_article, cluster(newsid)
test gpt35=gpt4
local p1 `r(p)'
local p1=round(`p1',0.001)
test gpt35=claude
local p2 `r(p)'
local p2=round(`p2',0.001)
test gpt4=sonet
local p3 `r(p)'
local p3=round(`p3',0.001)
outreg2 using "`tables_dir'/Tab_C1_main_reg.tex", append label symbol(***, **, *) tex(frag) ///
addtext(p-value: $\beta_{GPT3.5}=\beta_{GPT4}$, `p1', p-value: $\beta_{GPT3.5}=\beta_{Claude}$, `p2', p-value: $\beta_GPT4=\beta_{Sonet}$, `p3') ///
cons nonotes nor2
	
preserve
import delimited "`tables_dir'/Tab_C1_main_reg.tex", clear 
replace v1=subinstr(v1,"VARIABLES","",.)
sleep 100
export delimited "`tables_dir'/Tab_C1_main_reg.tex", replace novarn
restore


// Regression Table C2

*We explore the interaction with task difficulty in the two task where GPT3.5 does not outperform Qualtrics, obtaining a positive effect for the difficult news

gen interaction=gpt35*difficult_q4
label var interaction "GPT-3.5#Difficult"
replace difficult=difficult_q4

reg score_q4 gpt35 difficult interaction long_article if (gpt35==1 | qualtrics==1), cluster(newsid)
outreg2 using "`tables_dir'/Tab_C2_gpt35_difficult.tex", replace label symbol(***, **, *) tex(frag) ///
cons nonotes nor2


replace interaction=gpt35*difficult_any
replace difficult=difficult_any
	
reg score_all_perfect gpt35 difficult interaction  long_article if (gpt35==1 | qualtrics==1), cluster(newsid)
outreg2 using "`tables_dir'/Tab_C2_gpt35_difficult.tex", append label symbol(***, **, *) tex(frag) ///
cons nonotes nor2

preserve
import delimited "`tables_dir'/Tab_C2_gpt35_difficult.tex", clear 
replace v1=subinstr(v1,"VARIABLES","",.)
sleep 100
export delimited "`tables_dir'/Tab_C2_gpt35_difficult.tex", replace novarn
restore

// Regression Table C3

* We explore dynamic performance in human coders. 

replace difficult=difficult_any

reg score_all_perfect difficult long_article i.q_block_number if (qualtrics==1), cluster(newsid)
outreg2 using "`tables_dir'/Tab_C3_reg_qualtrics.tex", replace label symbol(***, **, *) tex(frag) nocons ///
addtext(Participant FE, NO) ///
nonotes nor2
	
reg score_all_perfect difficult long_article i.q_block_number i.participant_id if (qualtrics==1), cluster(newsid)
outreg2 using "`tables_dir'/Tab_C3_reg_qualtrics.tex", append label symbol(***, **, *) tex(frag) nocons ///
keep(score_all_perfect difficult long_article i.q_block_number) ///
addtext(Participant FE, YES) ///
nonotes nor2
	
preserve
import delimited "`tables_dir'/Tab_C3_reg_qualtrics.tex", clear 
replace v1=subinstr(v1,"VARIABLES","",.)
replace v1=subinstr(v1,"Q\_block\_number = 2","Second Article",.)
replace v1=subinstr(v1,"Q\_block\_number = 3","Third Article",.)
sleep 100
export delimited "`tables_dir'/Tab_C3_reg_qualtrics.tex", replace novarn
restore



// Table C6 

* We explore political bias
merge n:1 newsid using "../data/political_party_mayor.dta"
keep if _merge == 3
drop _merge

*-----------------------------------------------------*
* 2. Construct LLM-based classification
*-----------------------------------------------------*

* Criticism to right (PP)
gen LLM_criticismtoright = (coder_q5 == 5)
replace LLM_criticismtoright = 1 if coder_q5 == 3
replace LLM_criticismtoright = 1 if coder_q5 == 1 & party_mayor == "PP"
replace LLM_criticismtoright = 1 if coder_q5 == 4 & party_mayor == "PSOE"
replace LLM_criticismtoright = 0 if missing(LLM_criticismtoright)

* Criticism to left (PSOE)
gen LLM_criticismtoleft = (coder_q5 == 6)
replace LLM_criticismtoleft = 1 if coder_q5 == 1 & party_mayor == "PSOE"
replace LLM_criticismtoleft = 1 if coder_q5 == 4 & party_mayor == "PP"
replace LLM_criticismtoleft = 0 if missing(LLM_criticismtoleft)


*-----------------------------------------------------*
* 3. Construct gold-standard classification
*-----------------------------------------------------*

* Criticism to right (PP)
gen gold_criticismtoright = inlist(gold_q5, "{'5'}", "{'2', '5'}", "{'1', '5'}", "{'5', '1'}", "{'5', '2'}")
codebook gold_criticismtoright
replace gold_criticismtoright = 1 if gold_q5 == "{'1'}" & party_mayor == "PP"
replace gold_criticismtoright = 1 if gold_q5 == "{'3'}"
replace gold_criticismtoright = 1 if gold_q5 == "{'4'}" & party_mayor == "PSOE"
replace gold_criticismtoright = 0 if missing(gold_criticismtoright)



* Criticism to left (PSOE)
gen gold_criticismtoleft = inlist(gold_q5, "{'6'}", "{'2', '6'}", "{'1', '6'}", "{'2', '1', '6'}")
replace gold_criticismtoleft = 1 if gold_q5 == "{'1'}" & party_mayor == "PSOE"
replace gold_criticismtoleft = 1 if gold_q5 == "{'4'}" & party_mayor == "PP"
replace gold_criticismtoleft = 0 if missing(gold_criticismtoleft)

*-----------------------------------------------------*
* 4. Compute error rates
*-----------------------------------------------------*

* Type I error (false positives)
tab LLM_criticismtoright score_q5 if coder != "Q", matcell(F)
scalar a11 = F[2,1] / (F[2,1] + F[1,2])

tab LLM_criticismtoleft score_q5 if coder != "Q", matcell(F)
scalar a21 = F[2,1] / (F[2,1] + F[1,2])

* Type II error (false negatives)
tab gold_criticismtoright score_q5 if coder != "Q", matcell(F)
scalar a12 = F[2,1] / (F[2,1] + F[2,2])

tab gold_criticismtoleft score_q5 if coder != "Q", matcell(F)
scalar a22 = F[2,1] / (F[2,1] + F[2,2])

*-----------------------------------------------------*
* 5. Construct matrix
*-----------------------------------------------------*

matrix T = (a11, a12 \ a21, a22)
matrix rownames T = "Party = PP" "Party = PSOE"
matrix colnames T = "Type I Error" "Type II Error"

*-----------------------------------------------------*
* 6. Export results (Excel + LaTeX)
*-----------------------------------------------------*

* Excel
putexcel set "`tables_dir'/table_political_bias.xlsx", replace
putexcel A1 = matrix(T), names

* LaTeX (requires estout installed)
esttab matrix(T) using "`tables_dir'/table_political_bias.tex", replace ///
    booktabs nomtitles nonumbers alignment(lcc) ///
    cells("T(fmt(3))") ///
    title("Political Bias: Type I and Type II Errors")

*******************************************************
* End of file
*******************************************************


