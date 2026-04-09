#include "pose.h"

namespace {
#define DIM 18
#define EDIM 18
#define MEDIM 18
typedef void (*Hfun)(double *, double *, double *);
const static double MAHA_THRESH_4 = 7.814727903251177;
const static double MAHA_THRESH_10 = 7.814727903251177;
const static double MAHA_THRESH_13 = 7.814727903251177;
const static double MAHA_THRESH_14 = 7.814727903251177;

/******************************************************************************
 *                      Code generated with SymPy 1.14.0                      *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_6757027054840770867) {
   out_6757027054840770867[0] = delta_x[0] + nom_x[0];
   out_6757027054840770867[1] = delta_x[1] + nom_x[1];
   out_6757027054840770867[2] = delta_x[2] + nom_x[2];
   out_6757027054840770867[3] = delta_x[3] + nom_x[3];
   out_6757027054840770867[4] = delta_x[4] + nom_x[4];
   out_6757027054840770867[5] = delta_x[5] + nom_x[5];
   out_6757027054840770867[6] = delta_x[6] + nom_x[6];
   out_6757027054840770867[7] = delta_x[7] + nom_x[7];
   out_6757027054840770867[8] = delta_x[8] + nom_x[8];
   out_6757027054840770867[9] = delta_x[9] + nom_x[9];
   out_6757027054840770867[10] = delta_x[10] + nom_x[10];
   out_6757027054840770867[11] = delta_x[11] + nom_x[11];
   out_6757027054840770867[12] = delta_x[12] + nom_x[12];
   out_6757027054840770867[13] = delta_x[13] + nom_x[13];
   out_6757027054840770867[14] = delta_x[14] + nom_x[14];
   out_6757027054840770867[15] = delta_x[15] + nom_x[15];
   out_6757027054840770867[16] = delta_x[16] + nom_x[16];
   out_6757027054840770867[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_3041472167094278981) {
   out_3041472167094278981[0] = -nom_x[0] + true_x[0];
   out_3041472167094278981[1] = -nom_x[1] + true_x[1];
   out_3041472167094278981[2] = -nom_x[2] + true_x[2];
   out_3041472167094278981[3] = -nom_x[3] + true_x[3];
   out_3041472167094278981[4] = -nom_x[4] + true_x[4];
   out_3041472167094278981[5] = -nom_x[5] + true_x[5];
   out_3041472167094278981[6] = -nom_x[6] + true_x[6];
   out_3041472167094278981[7] = -nom_x[7] + true_x[7];
   out_3041472167094278981[8] = -nom_x[8] + true_x[8];
   out_3041472167094278981[9] = -nom_x[9] + true_x[9];
   out_3041472167094278981[10] = -nom_x[10] + true_x[10];
   out_3041472167094278981[11] = -nom_x[11] + true_x[11];
   out_3041472167094278981[12] = -nom_x[12] + true_x[12];
   out_3041472167094278981[13] = -nom_x[13] + true_x[13];
   out_3041472167094278981[14] = -nom_x[14] + true_x[14];
   out_3041472167094278981[15] = -nom_x[15] + true_x[15];
   out_3041472167094278981[16] = -nom_x[16] + true_x[16];
   out_3041472167094278981[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_2705022079487018887) {
   out_2705022079487018887[0] = 1.0;
   out_2705022079487018887[1] = 0.0;
   out_2705022079487018887[2] = 0.0;
   out_2705022079487018887[3] = 0.0;
   out_2705022079487018887[4] = 0.0;
   out_2705022079487018887[5] = 0.0;
   out_2705022079487018887[6] = 0.0;
   out_2705022079487018887[7] = 0.0;
   out_2705022079487018887[8] = 0.0;
   out_2705022079487018887[9] = 0.0;
   out_2705022079487018887[10] = 0.0;
   out_2705022079487018887[11] = 0.0;
   out_2705022079487018887[12] = 0.0;
   out_2705022079487018887[13] = 0.0;
   out_2705022079487018887[14] = 0.0;
   out_2705022079487018887[15] = 0.0;
   out_2705022079487018887[16] = 0.0;
   out_2705022079487018887[17] = 0.0;
   out_2705022079487018887[18] = 0.0;
   out_2705022079487018887[19] = 1.0;
   out_2705022079487018887[20] = 0.0;
   out_2705022079487018887[21] = 0.0;
   out_2705022079487018887[22] = 0.0;
   out_2705022079487018887[23] = 0.0;
   out_2705022079487018887[24] = 0.0;
   out_2705022079487018887[25] = 0.0;
   out_2705022079487018887[26] = 0.0;
   out_2705022079487018887[27] = 0.0;
   out_2705022079487018887[28] = 0.0;
   out_2705022079487018887[29] = 0.0;
   out_2705022079487018887[30] = 0.0;
   out_2705022079487018887[31] = 0.0;
   out_2705022079487018887[32] = 0.0;
   out_2705022079487018887[33] = 0.0;
   out_2705022079487018887[34] = 0.0;
   out_2705022079487018887[35] = 0.0;
   out_2705022079487018887[36] = 0.0;
   out_2705022079487018887[37] = 0.0;
   out_2705022079487018887[38] = 1.0;
   out_2705022079487018887[39] = 0.0;
   out_2705022079487018887[40] = 0.0;
   out_2705022079487018887[41] = 0.0;
   out_2705022079487018887[42] = 0.0;
   out_2705022079487018887[43] = 0.0;
   out_2705022079487018887[44] = 0.0;
   out_2705022079487018887[45] = 0.0;
   out_2705022079487018887[46] = 0.0;
   out_2705022079487018887[47] = 0.0;
   out_2705022079487018887[48] = 0.0;
   out_2705022079487018887[49] = 0.0;
   out_2705022079487018887[50] = 0.0;
   out_2705022079487018887[51] = 0.0;
   out_2705022079487018887[52] = 0.0;
   out_2705022079487018887[53] = 0.0;
   out_2705022079487018887[54] = 0.0;
   out_2705022079487018887[55] = 0.0;
   out_2705022079487018887[56] = 0.0;
   out_2705022079487018887[57] = 1.0;
   out_2705022079487018887[58] = 0.0;
   out_2705022079487018887[59] = 0.0;
   out_2705022079487018887[60] = 0.0;
   out_2705022079487018887[61] = 0.0;
   out_2705022079487018887[62] = 0.0;
   out_2705022079487018887[63] = 0.0;
   out_2705022079487018887[64] = 0.0;
   out_2705022079487018887[65] = 0.0;
   out_2705022079487018887[66] = 0.0;
   out_2705022079487018887[67] = 0.0;
   out_2705022079487018887[68] = 0.0;
   out_2705022079487018887[69] = 0.0;
   out_2705022079487018887[70] = 0.0;
   out_2705022079487018887[71] = 0.0;
   out_2705022079487018887[72] = 0.0;
   out_2705022079487018887[73] = 0.0;
   out_2705022079487018887[74] = 0.0;
   out_2705022079487018887[75] = 0.0;
   out_2705022079487018887[76] = 1.0;
   out_2705022079487018887[77] = 0.0;
   out_2705022079487018887[78] = 0.0;
   out_2705022079487018887[79] = 0.0;
   out_2705022079487018887[80] = 0.0;
   out_2705022079487018887[81] = 0.0;
   out_2705022079487018887[82] = 0.0;
   out_2705022079487018887[83] = 0.0;
   out_2705022079487018887[84] = 0.0;
   out_2705022079487018887[85] = 0.0;
   out_2705022079487018887[86] = 0.0;
   out_2705022079487018887[87] = 0.0;
   out_2705022079487018887[88] = 0.0;
   out_2705022079487018887[89] = 0.0;
   out_2705022079487018887[90] = 0.0;
   out_2705022079487018887[91] = 0.0;
   out_2705022079487018887[92] = 0.0;
   out_2705022079487018887[93] = 0.0;
   out_2705022079487018887[94] = 0.0;
   out_2705022079487018887[95] = 1.0;
   out_2705022079487018887[96] = 0.0;
   out_2705022079487018887[97] = 0.0;
   out_2705022079487018887[98] = 0.0;
   out_2705022079487018887[99] = 0.0;
   out_2705022079487018887[100] = 0.0;
   out_2705022079487018887[101] = 0.0;
   out_2705022079487018887[102] = 0.0;
   out_2705022079487018887[103] = 0.0;
   out_2705022079487018887[104] = 0.0;
   out_2705022079487018887[105] = 0.0;
   out_2705022079487018887[106] = 0.0;
   out_2705022079487018887[107] = 0.0;
   out_2705022079487018887[108] = 0.0;
   out_2705022079487018887[109] = 0.0;
   out_2705022079487018887[110] = 0.0;
   out_2705022079487018887[111] = 0.0;
   out_2705022079487018887[112] = 0.0;
   out_2705022079487018887[113] = 0.0;
   out_2705022079487018887[114] = 1.0;
   out_2705022079487018887[115] = 0.0;
   out_2705022079487018887[116] = 0.0;
   out_2705022079487018887[117] = 0.0;
   out_2705022079487018887[118] = 0.0;
   out_2705022079487018887[119] = 0.0;
   out_2705022079487018887[120] = 0.0;
   out_2705022079487018887[121] = 0.0;
   out_2705022079487018887[122] = 0.0;
   out_2705022079487018887[123] = 0.0;
   out_2705022079487018887[124] = 0.0;
   out_2705022079487018887[125] = 0.0;
   out_2705022079487018887[126] = 0.0;
   out_2705022079487018887[127] = 0.0;
   out_2705022079487018887[128] = 0.0;
   out_2705022079487018887[129] = 0.0;
   out_2705022079487018887[130] = 0.0;
   out_2705022079487018887[131] = 0.0;
   out_2705022079487018887[132] = 0.0;
   out_2705022079487018887[133] = 1.0;
   out_2705022079487018887[134] = 0.0;
   out_2705022079487018887[135] = 0.0;
   out_2705022079487018887[136] = 0.0;
   out_2705022079487018887[137] = 0.0;
   out_2705022079487018887[138] = 0.0;
   out_2705022079487018887[139] = 0.0;
   out_2705022079487018887[140] = 0.0;
   out_2705022079487018887[141] = 0.0;
   out_2705022079487018887[142] = 0.0;
   out_2705022079487018887[143] = 0.0;
   out_2705022079487018887[144] = 0.0;
   out_2705022079487018887[145] = 0.0;
   out_2705022079487018887[146] = 0.0;
   out_2705022079487018887[147] = 0.0;
   out_2705022079487018887[148] = 0.0;
   out_2705022079487018887[149] = 0.0;
   out_2705022079487018887[150] = 0.0;
   out_2705022079487018887[151] = 0.0;
   out_2705022079487018887[152] = 1.0;
   out_2705022079487018887[153] = 0.0;
   out_2705022079487018887[154] = 0.0;
   out_2705022079487018887[155] = 0.0;
   out_2705022079487018887[156] = 0.0;
   out_2705022079487018887[157] = 0.0;
   out_2705022079487018887[158] = 0.0;
   out_2705022079487018887[159] = 0.0;
   out_2705022079487018887[160] = 0.0;
   out_2705022079487018887[161] = 0.0;
   out_2705022079487018887[162] = 0.0;
   out_2705022079487018887[163] = 0.0;
   out_2705022079487018887[164] = 0.0;
   out_2705022079487018887[165] = 0.0;
   out_2705022079487018887[166] = 0.0;
   out_2705022079487018887[167] = 0.0;
   out_2705022079487018887[168] = 0.0;
   out_2705022079487018887[169] = 0.0;
   out_2705022079487018887[170] = 0.0;
   out_2705022079487018887[171] = 1.0;
   out_2705022079487018887[172] = 0.0;
   out_2705022079487018887[173] = 0.0;
   out_2705022079487018887[174] = 0.0;
   out_2705022079487018887[175] = 0.0;
   out_2705022079487018887[176] = 0.0;
   out_2705022079487018887[177] = 0.0;
   out_2705022079487018887[178] = 0.0;
   out_2705022079487018887[179] = 0.0;
   out_2705022079487018887[180] = 0.0;
   out_2705022079487018887[181] = 0.0;
   out_2705022079487018887[182] = 0.0;
   out_2705022079487018887[183] = 0.0;
   out_2705022079487018887[184] = 0.0;
   out_2705022079487018887[185] = 0.0;
   out_2705022079487018887[186] = 0.0;
   out_2705022079487018887[187] = 0.0;
   out_2705022079487018887[188] = 0.0;
   out_2705022079487018887[189] = 0.0;
   out_2705022079487018887[190] = 1.0;
   out_2705022079487018887[191] = 0.0;
   out_2705022079487018887[192] = 0.0;
   out_2705022079487018887[193] = 0.0;
   out_2705022079487018887[194] = 0.0;
   out_2705022079487018887[195] = 0.0;
   out_2705022079487018887[196] = 0.0;
   out_2705022079487018887[197] = 0.0;
   out_2705022079487018887[198] = 0.0;
   out_2705022079487018887[199] = 0.0;
   out_2705022079487018887[200] = 0.0;
   out_2705022079487018887[201] = 0.0;
   out_2705022079487018887[202] = 0.0;
   out_2705022079487018887[203] = 0.0;
   out_2705022079487018887[204] = 0.0;
   out_2705022079487018887[205] = 0.0;
   out_2705022079487018887[206] = 0.0;
   out_2705022079487018887[207] = 0.0;
   out_2705022079487018887[208] = 0.0;
   out_2705022079487018887[209] = 1.0;
   out_2705022079487018887[210] = 0.0;
   out_2705022079487018887[211] = 0.0;
   out_2705022079487018887[212] = 0.0;
   out_2705022079487018887[213] = 0.0;
   out_2705022079487018887[214] = 0.0;
   out_2705022079487018887[215] = 0.0;
   out_2705022079487018887[216] = 0.0;
   out_2705022079487018887[217] = 0.0;
   out_2705022079487018887[218] = 0.0;
   out_2705022079487018887[219] = 0.0;
   out_2705022079487018887[220] = 0.0;
   out_2705022079487018887[221] = 0.0;
   out_2705022079487018887[222] = 0.0;
   out_2705022079487018887[223] = 0.0;
   out_2705022079487018887[224] = 0.0;
   out_2705022079487018887[225] = 0.0;
   out_2705022079487018887[226] = 0.0;
   out_2705022079487018887[227] = 0.0;
   out_2705022079487018887[228] = 1.0;
   out_2705022079487018887[229] = 0.0;
   out_2705022079487018887[230] = 0.0;
   out_2705022079487018887[231] = 0.0;
   out_2705022079487018887[232] = 0.0;
   out_2705022079487018887[233] = 0.0;
   out_2705022079487018887[234] = 0.0;
   out_2705022079487018887[235] = 0.0;
   out_2705022079487018887[236] = 0.0;
   out_2705022079487018887[237] = 0.0;
   out_2705022079487018887[238] = 0.0;
   out_2705022079487018887[239] = 0.0;
   out_2705022079487018887[240] = 0.0;
   out_2705022079487018887[241] = 0.0;
   out_2705022079487018887[242] = 0.0;
   out_2705022079487018887[243] = 0.0;
   out_2705022079487018887[244] = 0.0;
   out_2705022079487018887[245] = 0.0;
   out_2705022079487018887[246] = 0.0;
   out_2705022079487018887[247] = 1.0;
   out_2705022079487018887[248] = 0.0;
   out_2705022079487018887[249] = 0.0;
   out_2705022079487018887[250] = 0.0;
   out_2705022079487018887[251] = 0.0;
   out_2705022079487018887[252] = 0.0;
   out_2705022079487018887[253] = 0.0;
   out_2705022079487018887[254] = 0.0;
   out_2705022079487018887[255] = 0.0;
   out_2705022079487018887[256] = 0.0;
   out_2705022079487018887[257] = 0.0;
   out_2705022079487018887[258] = 0.0;
   out_2705022079487018887[259] = 0.0;
   out_2705022079487018887[260] = 0.0;
   out_2705022079487018887[261] = 0.0;
   out_2705022079487018887[262] = 0.0;
   out_2705022079487018887[263] = 0.0;
   out_2705022079487018887[264] = 0.0;
   out_2705022079487018887[265] = 0.0;
   out_2705022079487018887[266] = 1.0;
   out_2705022079487018887[267] = 0.0;
   out_2705022079487018887[268] = 0.0;
   out_2705022079487018887[269] = 0.0;
   out_2705022079487018887[270] = 0.0;
   out_2705022079487018887[271] = 0.0;
   out_2705022079487018887[272] = 0.0;
   out_2705022079487018887[273] = 0.0;
   out_2705022079487018887[274] = 0.0;
   out_2705022079487018887[275] = 0.0;
   out_2705022079487018887[276] = 0.0;
   out_2705022079487018887[277] = 0.0;
   out_2705022079487018887[278] = 0.0;
   out_2705022079487018887[279] = 0.0;
   out_2705022079487018887[280] = 0.0;
   out_2705022079487018887[281] = 0.0;
   out_2705022079487018887[282] = 0.0;
   out_2705022079487018887[283] = 0.0;
   out_2705022079487018887[284] = 0.0;
   out_2705022079487018887[285] = 1.0;
   out_2705022079487018887[286] = 0.0;
   out_2705022079487018887[287] = 0.0;
   out_2705022079487018887[288] = 0.0;
   out_2705022079487018887[289] = 0.0;
   out_2705022079487018887[290] = 0.0;
   out_2705022079487018887[291] = 0.0;
   out_2705022079487018887[292] = 0.0;
   out_2705022079487018887[293] = 0.0;
   out_2705022079487018887[294] = 0.0;
   out_2705022079487018887[295] = 0.0;
   out_2705022079487018887[296] = 0.0;
   out_2705022079487018887[297] = 0.0;
   out_2705022079487018887[298] = 0.0;
   out_2705022079487018887[299] = 0.0;
   out_2705022079487018887[300] = 0.0;
   out_2705022079487018887[301] = 0.0;
   out_2705022079487018887[302] = 0.0;
   out_2705022079487018887[303] = 0.0;
   out_2705022079487018887[304] = 1.0;
   out_2705022079487018887[305] = 0.0;
   out_2705022079487018887[306] = 0.0;
   out_2705022079487018887[307] = 0.0;
   out_2705022079487018887[308] = 0.0;
   out_2705022079487018887[309] = 0.0;
   out_2705022079487018887[310] = 0.0;
   out_2705022079487018887[311] = 0.0;
   out_2705022079487018887[312] = 0.0;
   out_2705022079487018887[313] = 0.0;
   out_2705022079487018887[314] = 0.0;
   out_2705022079487018887[315] = 0.0;
   out_2705022079487018887[316] = 0.0;
   out_2705022079487018887[317] = 0.0;
   out_2705022079487018887[318] = 0.0;
   out_2705022079487018887[319] = 0.0;
   out_2705022079487018887[320] = 0.0;
   out_2705022079487018887[321] = 0.0;
   out_2705022079487018887[322] = 0.0;
   out_2705022079487018887[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_5934707486323140053) {
   out_5934707486323140053[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_5934707486323140053[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_5934707486323140053[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_5934707486323140053[3] = dt*state[12] + state[3];
   out_5934707486323140053[4] = dt*state[13] + state[4];
   out_5934707486323140053[5] = dt*state[14] + state[5];
   out_5934707486323140053[6] = state[6];
   out_5934707486323140053[7] = state[7];
   out_5934707486323140053[8] = state[8];
   out_5934707486323140053[9] = state[9];
   out_5934707486323140053[10] = state[10];
   out_5934707486323140053[11] = state[11];
   out_5934707486323140053[12] = state[12];
   out_5934707486323140053[13] = state[13];
   out_5934707486323140053[14] = state[14];
   out_5934707486323140053[15] = state[15];
   out_5934707486323140053[16] = state[16];
   out_5934707486323140053[17] = state[17];
}
void F_fun(double *state, double dt, double *out_6052904124511391966) {
   out_6052904124511391966[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_6052904124511391966[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_6052904124511391966[2] = 0;
   out_6052904124511391966[3] = 0;
   out_6052904124511391966[4] = 0;
   out_6052904124511391966[5] = 0;
   out_6052904124511391966[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_6052904124511391966[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_6052904124511391966[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_6052904124511391966[9] = 0;
   out_6052904124511391966[10] = 0;
   out_6052904124511391966[11] = 0;
   out_6052904124511391966[12] = 0;
   out_6052904124511391966[13] = 0;
   out_6052904124511391966[14] = 0;
   out_6052904124511391966[15] = 0;
   out_6052904124511391966[16] = 0;
   out_6052904124511391966[17] = 0;
   out_6052904124511391966[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_6052904124511391966[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_6052904124511391966[20] = 0;
   out_6052904124511391966[21] = 0;
   out_6052904124511391966[22] = 0;
   out_6052904124511391966[23] = 0;
   out_6052904124511391966[24] = 0;
   out_6052904124511391966[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_6052904124511391966[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_6052904124511391966[27] = 0;
   out_6052904124511391966[28] = 0;
   out_6052904124511391966[29] = 0;
   out_6052904124511391966[30] = 0;
   out_6052904124511391966[31] = 0;
   out_6052904124511391966[32] = 0;
   out_6052904124511391966[33] = 0;
   out_6052904124511391966[34] = 0;
   out_6052904124511391966[35] = 0;
   out_6052904124511391966[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_6052904124511391966[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_6052904124511391966[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_6052904124511391966[39] = 0;
   out_6052904124511391966[40] = 0;
   out_6052904124511391966[41] = 0;
   out_6052904124511391966[42] = 0;
   out_6052904124511391966[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_6052904124511391966[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_6052904124511391966[45] = 0;
   out_6052904124511391966[46] = 0;
   out_6052904124511391966[47] = 0;
   out_6052904124511391966[48] = 0;
   out_6052904124511391966[49] = 0;
   out_6052904124511391966[50] = 0;
   out_6052904124511391966[51] = 0;
   out_6052904124511391966[52] = 0;
   out_6052904124511391966[53] = 0;
   out_6052904124511391966[54] = 0;
   out_6052904124511391966[55] = 0;
   out_6052904124511391966[56] = 0;
   out_6052904124511391966[57] = 1;
   out_6052904124511391966[58] = 0;
   out_6052904124511391966[59] = 0;
   out_6052904124511391966[60] = 0;
   out_6052904124511391966[61] = 0;
   out_6052904124511391966[62] = 0;
   out_6052904124511391966[63] = 0;
   out_6052904124511391966[64] = 0;
   out_6052904124511391966[65] = 0;
   out_6052904124511391966[66] = dt;
   out_6052904124511391966[67] = 0;
   out_6052904124511391966[68] = 0;
   out_6052904124511391966[69] = 0;
   out_6052904124511391966[70] = 0;
   out_6052904124511391966[71] = 0;
   out_6052904124511391966[72] = 0;
   out_6052904124511391966[73] = 0;
   out_6052904124511391966[74] = 0;
   out_6052904124511391966[75] = 0;
   out_6052904124511391966[76] = 1;
   out_6052904124511391966[77] = 0;
   out_6052904124511391966[78] = 0;
   out_6052904124511391966[79] = 0;
   out_6052904124511391966[80] = 0;
   out_6052904124511391966[81] = 0;
   out_6052904124511391966[82] = 0;
   out_6052904124511391966[83] = 0;
   out_6052904124511391966[84] = 0;
   out_6052904124511391966[85] = dt;
   out_6052904124511391966[86] = 0;
   out_6052904124511391966[87] = 0;
   out_6052904124511391966[88] = 0;
   out_6052904124511391966[89] = 0;
   out_6052904124511391966[90] = 0;
   out_6052904124511391966[91] = 0;
   out_6052904124511391966[92] = 0;
   out_6052904124511391966[93] = 0;
   out_6052904124511391966[94] = 0;
   out_6052904124511391966[95] = 1;
   out_6052904124511391966[96] = 0;
   out_6052904124511391966[97] = 0;
   out_6052904124511391966[98] = 0;
   out_6052904124511391966[99] = 0;
   out_6052904124511391966[100] = 0;
   out_6052904124511391966[101] = 0;
   out_6052904124511391966[102] = 0;
   out_6052904124511391966[103] = 0;
   out_6052904124511391966[104] = dt;
   out_6052904124511391966[105] = 0;
   out_6052904124511391966[106] = 0;
   out_6052904124511391966[107] = 0;
   out_6052904124511391966[108] = 0;
   out_6052904124511391966[109] = 0;
   out_6052904124511391966[110] = 0;
   out_6052904124511391966[111] = 0;
   out_6052904124511391966[112] = 0;
   out_6052904124511391966[113] = 0;
   out_6052904124511391966[114] = 1;
   out_6052904124511391966[115] = 0;
   out_6052904124511391966[116] = 0;
   out_6052904124511391966[117] = 0;
   out_6052904124511391966[118] = 0;
   out_6052904124511391966[119] = 0;
   out_6052904124511391966[120] = 0;
   out_6052904124511391966[121] = 0;
   out_6052904124511391966[122] = 0;
   out_6052904124511391966[123] = 0;
   out_6052904124511391966[124] = 0;
   out_6052904124511391966[125] = 0;
   out_6052904124511391966[126] = 0;
   out_6052904124511391966[127] = 0;
   out_6052904124511391966[128] = 0;
   out_6052904124511391966[129] = 0;
   out_6052904124511391966[130] = 0;
   out_6052904124511391966[131] = 0;
   out_6052904124511391966[132] = 0;
   out_6052904124511391966[133] = 1;
   out_6052904124511391966[134] = 0;
   out_6052904124511391966[135] = 0;
   out_6052904124511391966[136] = 0;
   out_6052904124511391966[137] = 0;
   out_6052904124511391966[138] = 0;
   out_6052904124511391966[139] = 0;
   out_6052904124511391966[140] = 0;
   out_6052904124511391966[141] = 0;
   out_6052904124511391966[142] = 0;
   out_6052904124511391966[143] = 0;
   out_6052904124511391966[144] = 0;
   out_6052904124511391966[145] = 0;
   out_6052904124511391966[146] = 0;
   out_6052904124511391966[147] = 0;
   out_6052904124511391966[148] = 0;
   out_6052904124511391966[149] = 0;
   out_6052904124511391966[150] = 0;
   out_6052904124511391966[151] = 0;
   out_6052904124511391966[152] = 1;
   out_6052904124511391966[153] = 0;
   out_6052904124511391966[154] = 0;
   out_6052904124511391966[155] = 0;
   out_6052904124511391966[156] = 0;
   out_6052904124511391966[157] = 0;
   out_6052904124511391966[158] = 0;
   out_6052904124511391966[159] = 0;
   out_6052904124511391966[160] = 0;
   out_6052904124511391966[161] = 0;
   out_6052904124511391966[162] = 0;
   out_6052904124511391966[163] = 0;
   out_6052904124511391966[164] = 0;
   out_6052904124511391966[165] = 0;
   out_6052904124511391966[166] = 0;
   out_6052904124511391966[167] = 0;
   out_6052904124511391966[168] = 0;
   out_6052904124511391966[169] = 0;
   out_6052904124511391966[170] = 0;
   out_6052904124511391966[171] = 1;
   out_6052904124511391966[172] = 0;
   out_6052904124511391966[173] = 0;
   out_6052904124511391966[174] = 0;
   out_6052904124511391966[175] = 0;
   out_6052904124511391966[176] = 0;
   out_6052904124511391966[177] = 0;
   out_6052904124511391966[178] = 0;
   out_6052904124511391966[179] = 0;
   out_6052904124511391966[180] = 0;
   out_6052904124511391966[181] = 0;
   out_6052904124511391966[182] = 0;
   out_6052904124511391966[183] = 0;
   out_6052904124511391966[184] = 0;
   out_6052904124511391966[185] = 0;
   out_6052904124511391966[186] = 0;
   out_6052904124511391966[187] = 0;
   out_6052904124511391966[188] = 0;
   out_6052904124511391966[189] = 0;
   out_6052904124511391966[190] = 1;
   out_6052904124511391966[191] = 0;
   out_6052904124511391966[192] = 0;
   out_6052904124511391966[193] = 0;
   out_6052904124511391966[194] = 0;
   out_6052904124511391966[195] = 0;
   out_6052904124511391966[196] = 0;
   out_6052904124511391966[197] = 0;
   out_6052904124511391966[198] = 0;
   out_6052904124511391966[199] = 0;
   out_6052904124511391966[200] = 0;
   out_6052904124511391966[201] = 0;
   out_6052904124511391966[202] = 0;
   out_6052904124511391966[203] = 0;
   out_6052904124511391966[204] = 0;
   out_6052904124511391966[205] = 0;
   out_6052904124511391966[206] = 0;
   out_6052904124511391966[207] = 0;
   out_6052904124511391966[208] = 0;
   out_6052904124511391966[209] = 1;
   out_6052904124511391966[210] = 0;
   out_6052904124511391966[211] = 0;
   out_6052904124511391966[212] = 0;
   out_6052904124511391966[213] = 0;
   out_6052904124511391966[214] = 0;
   out_6052904124511391966[215] = 0;
   out_6052904124511391966[216] = 0;
   out_6052904124511391966[217] = 0;
   out_6052904124511391966[218] = 0;
   out_6052904124511391966[219] = 0;
   out_6052904124511391966[220] = 0;
   out_6052904124511391966[221] = 0;
   out_6052904124511391966[222] = 0;
   out_6052904124511391966[223] = 0;
   out_6052904124511391966[224] = 0;
   out_6052904124511391966[225] = 0;
   out_6052904124511391966[226] = 0;
   out_6052904124511391966[227] = 0;
   out_6052904124511391966[228] = 1;
   out_6052904124511391966[229] = 0;
   out_6052904124511391966[230] = 0;
   out_6052904124511391966[231] = 0;
   out_6052904124511391966[232] = 0;
   out_6052904124511391966[233] = 0;
   out_6052904124511391966[234] = 0;
   out_6052904124511391966[235] = 0;
   out_6052904124511391966[236] = 0;
   out_6052904124511391966[237] = 0;
   out_6052904124511391966[238] = 0;
   out_6052904124511391966[239] = 0;
   out_6052904124511391966[240] = 0;
   out_6052904124511391966[241] = 0;
   out_6052904124511391966[242] = 0;
   out_6052904124511391966[243] = 0;
   out_6052904124511391966[244] = 0;
   out_6052904124511391966[245] = 0;
   out_6052904124511391966[246] = 0;
   out_6052904124511391966[247] = 1;
   out_6052904124511391966[248] = 0;
   out_6052904124511391966[249] = 0;
   out_6052904124511391966[250] = 0;
   out_6052904124511391966[251] = 0;
   out_6052904124511391966[252] = 0;
   out_6052904124511391966[253] = 0;
   out_6052904124511391966[254] = 0;
   out_6052904124511391966[255] = 0;
   out_6052904124511391966[256] = 0;
   out_6052904124511391966[257] = 0;
   out_6052904124511391966[258] = 0;
   out_6052904124511391966[259] = 0;
   out_6052904124511391966[260] = 0;
   out_6052904124511391966[261] = 0;
   out_6052904124511391966[262] = 0;
   out_6052904124511391966[263] = 0;
   out_6052904124511391966[264] = 0;
   out_6052904124511391966[265] = 0;
   out_6052904124511391966[266] = 1;
   out_6052904124511391966[267] = 0;
   out_6052904124511391966[268] = 0;
   out_6052904124511391966[269] = 0;
   out_6052904124511391966[270] = 0;
   out_6052904124511391966[271] = 0;
   out_6052904124511391966[272] = 0;
   out_6052904124511391966[273] = 0;
   out_6052904124511391966[274] = 0;
   out_6052904124511391966[275] = 0;
   out_6052904124511391966[276] = 0;
   out_6052904124511391966[277] = 0;
   out_6052904124511391966[278] = 0;
   out_6052904124511391966[279] = 0;
   out_6052904124511391966[280] = 0;
   out_6052904124511391966[281] = 0;
   out_6052904124511391966[282] = 0;
   out_6052904124511391966[283] = 0;
   out_6052904124511391966[284] = 0;
   out_6052904124511391966[285] = 1;
   out_6052904124511391966[286] = 0;
   out_6052904124511391966[287] = 0;
   out_6052904124511391966[288] = 0;
   out_6052904124511391966[289] = 0;
   out_6052904124511391966[290] = 0;
   out_6052904124511391966[291] = 0;
   out_6052904124511391966[292] = 0;
   out_6052904124511391966[293] = 0;
   out_6052904124511391966[294] = 0;
   out_6052904124511391966[295] = 0;
   out_6052904124511391966[296] = 0;
   out_6052904124511391966[297] = 0;
   out_6052904124511391966[298] = 0;
   out_6052904124511391966[299] = 0;
   out_6052904124511391966[300] = 0;
   out_6052904124511391966[301] = 0;
   out_6052904124511391966[302] = 0;
   out_6052904124511391966[303] = 0;
   out_6052904124511391966[304] = 1;
   out_6052904124511391966[305] = 0;
   out_6052904124511391966[306] = 0;
   out_6052904124511391966[307] = 0;
   out_6052904124511391966[308] = 0;
   out_6052904124511391966[309] = 0;
   out_6052904124511391966[310] = 0;
   out_6052904124511391966[311] = 0;
   out_6052904124511391966[312] = 0;
   out_6052904124511391966[313] = 0;
   out_6052904124511391966[314] = 0;
   out_6052904124511391966[315] = 0;
   out_6052904124511391966[316] = 0;
   out_6052904124511391966[317] = 0;
   out_6052904124511391966[318] = 0;
   out_6052904124511391966[319] = 0;
   out_6052904124511391966[320] = 0;
   out_6052904124511391966[321] = 0;
   out_6052904124511391966[322] = 0;
   out_6052904124511391966[323] = 1;
}
void h_4(double *state, double *unused, double *out_2598800994798863224) {
   out_2598800994798863224[0] = state[6] + state[9];
   out_2598800994798863224[1] = state[7] + state[10];
   out_2598800994798863224[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_3143059727264939834) {
   out_3143059727264939834[0] = 0;
   out_3143059727264939834[1] = 0;
   out_3143059727264939834[2] = 0;
   out_3143059727264939834[3] = 0;
   out_3143059727264939834[4] = 0;
   out_3143059727264939834[5] = 0;
   out_3143059727264939834[6] = 1;
   out_3143059727264939834[7] = 0;
   out_3143059727264939834[8] = 0;
   out_3143059727264939834[9] = 1;
   out_3143059727264939834[10] = 0;
   out_3143059727264939834[11] = 0;
   out_3143059727264939834[12] = 0;
   out_3143059727264939834[13] = 0;
   out_3143059727264939834[14] = 0;
   out_3143059727264939834[15] = 0;
   out_3143059727264939834[16] = 0;
   out_3143059727264939834[17] = 0;
   out_3143059727264939834[18] = 0;
   out_3143059727264939834[19] = 0;
   out_3143059727264939834[20] = 0;
   out_3143059727264939834[21] = 0;
   out_3143059727264939834[22] = 0;
   out_3143059727264939834[23] = 0;
   out_3143059727264939834[24] = 0;
   out_3143059727264939834[25] = 1;
   out_3143059727264939834[26] = 0;
   out_3143059727264939834[27] = 0;
   out_3143059727264939834[28] = 1;
   out_3143059727264939834[29] = 0;
   out_3143059727264939834[30] = 0;
   out_3143059727264939834[31] = 0;
   out_3143059727264939834[32] = 0;
   out_3143059727264939834[33] = 0;
   out_3143059727264939834[34] = 0;
   out_3143059727264939834[35] = 0;
   out_3143059727264939834[36] = 0;
   out_3143059727264939834[37] = 0;
   out_3143059727264939834[38] = 0;
   out_3143059727264939834[39] = 0;
   out_3143059727264939834[40] = 0;
   out_3143059727264939834[41] = 0;
   out_3143059727264939834[42] = 0;
   out_3143059727264939834[43] = 0;
   out_3143059727264939834[44] = 1;
   out_3143059727264939834[45] = 0;
   out_3143059727264939834[46] = 0;
   out_3143059727264939834[47] = 1;
   out_3143059727264939834[48] = 0;
   out_3143059727264939834[49] = 0;
   out_3143059727264939834[50] = 0;
   out_3143059727264939834[51] = 0;
   out_3143059727264939834[52] = 0;
   out_3143059727264939834[53] = 0;
}
void h_10(double *state, double *unused, double *out_1573633236380714031) {
   out_1573633236380714031[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_1573633236380714031[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_1573633236380714031[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_7106231390164887414) {
   out_7106231390164887414[0] = 0;
   out_7106231390164887414[1] = 9.8100000000000005*cos(state[1]);
   out_7106231390164887414[2] = 0;
   out_7106231390164887414[3] = 0;
   out_7106231390164887414[4] = -state[8];
   out_7106231390164887414[5] = state[7];
   out_7106231390164887414[6] = 0;
   out_7106231390164887414[7] = state[5];
   out_7106231390164887414[8] = -state[4];
   out_7106231390164887414[9] = 0;
   out_7106231390164887414[10] = 0;
   out_7106231390164887414[11] = 0;
   out_7106231390164887414[12] = 1;
   out_7106231390164887414[13] = 0;
   out_7106231390164887414[14] = 0;
   out_7106231390164887414[15] = 1;
   out_7106231390164887414[16] = 0;
   out_7106231390164887414[17] = 0;
   out_7106231390164887414[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_7106231390164887414[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_7106231390164887414[20] = 0;
   out_7106231390164887414[21] = state[8];
   out_7106231390164887414[22] = 0;
   out_7106231390164887414[23] = -state[6];
   out_7106231390164887414[24] = -state[5];
   out_7106231390164887414[25] = 0;
   out_7106231390164887414[26] = state[3];
   out_7106231390164887414[27] = 0;
   out_7106231390164887414[28] = 0;
   out_7106231390164887414[29] = 0;
   out_7106231390164887414[30] = 0;
   out_7106231390164887414[31] = 1;
   out_7106231390164887414[32] = 0;
   out_7106231390164887414[33] = 0;
   out_7106231390164887414[34] = 1;
   out_7106231390164887414[35] = 0;
   out_7106231390164887414[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_7106231390164887414[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_7106231390164887414[38] = 0;
   out_7106231390164887414[39] = -state[7];
   out_7106231390164887414[40] = state[6];
   out_7106231390164887414[41] = 0;
   out_7106231390164887414[42] = state[4];
   out_7106231390164887414[43] = -state[3];
   out_7106231390164887414[44] = 0;
   out_7106231390164887414[45] = 0;
   out_7106231390164887414[46] = 0;
   out_7106231390164887414[47] = 0;
   out_7106231390164887414[48] = 0;
   out_7106231390164887414[49] = 0;
   out_7106231390164887414[50] = 1;
   out_7106231390164887414[51] = 0;
   out_7106231390164887414[52] = 0;
   out_7106231390164887414[53] = 1;
}
void h_13(double *state, double *unused, double *out_7607426681960106713) {
   out_7607426681960106713[0] = state[3];
   out_7607426681960106713[1] = state[4];
   out_7607426681960106713[2] = state[5];
}
void H_13(double *state, double *unused, double *out_69214098067392967) {
   out_69214098067392967[0] = 0;
   out_69214098067392967[1] = 0;
   out_69214098067392967[2] = 0;
   out_69214098067392967[3] = 1;
   out_69214098067392967[4] = 0;
   out_69214098067392967[5] = 0;
   out_69214098067392967[6] = 0;
   out_69214098067392967[7] = 0;
   out_69214098067392967[8] = 0;
   out_69214098067392967[9] = 0;
   out_69214098067392967[10] = 0;
   out_69214098067392967[11] = 0;
   out_69214098067392967[12] = 0;
   out_69214098067392967[13] = 0;
   out_69214098067392967[14] = 0;
   out_69214098067392967[15] = 0;
   out_69214098067392967[16] = 0;
   out_69214098067392967[17] = 0;
   out_69214098067392967[18] = 0;
   out_69214098067392967[19] = 0;
   out_69214098067392967[20] = 0;
   out_69214098067392967[21] = 0;
   out_69214098067392967[22] = 1;
   out_69214098067392967[23] = 0;
   out_69214098067392967[24] = 0;
   out_69214098067392967[25] = 0;
   out_69214098067392967[26] = 0;
   out_69214098067392967[27] = 0;
   out_69214098067392967[28] = 0;
   out_69214098067392967[29] = 0;
   out_69214098067392967[30] = 0;
   out_69214098067392967[31] = 0;
   out_69214098067392967[32] = 0;
   out_69214098067392967[33] = 0;
   out_69214098067392967[34] = 0;
   out_69214098067392967[35] = 0;
   out_69214098067392967[36] = 0;
   out_69214098067392967[37] = 0;
   out_69214098067392967[38] = 0;
   out_69214098067392967[39] = 0;
   out_69214098067392967[40] = 0;
   out_69214098067392967[41] = 1;
   out_69214098067392967[42] = 0;
   out_69214098067392967[43] = 0;
   out_69214098067392967[44] = 0;
   out_69214098067392967[45] = 0;
   out_69214098067392967[46] = 0;
   out_69214098067392967[47] = 0;
   out_69214098067392967[48] = 0;
   out_69214098067392967[49] = 0;
   out_69214098067392967[50] = 0;
   out_69214098067392967[51] = 0;
   out_69214098067392967[52] = 0;
   out_69214098067392967[53] = 0;
}
void h_14(double *state, double *unused, double *out_5439578415485324028) {
   out_5439578415485324028[0] = state[6];
   out_5439578415485324028[1] = state[7];
   out_5439578415485324028[2] = state[8];
}
void H_14(double *state, double *unused, double *out_820181129074544695) {
   out_820181129074544695[0] = 0;
   out_820181129074544695[1] = 0;
   out_820181129074544695[2] = 0;
   out_820181129074544695[3] = 0;
   out_820181129074544695[4] = 0;
   out_820181129074544695[5] = 0;
   out_820181129074544695[6] = 1;
   out_820181129074544695[7] = 0;
   out_820181129074544695[8] = 0;
   out_820181129074544695[9] = 0;
   out_820181129074544695[10] = 0;
   out_820181129074544695[11] = 0;
   out_820181129074544695[12] = 0;
   out_820181129074544695[13] = 0;
   out_820181129074544695[14] = 0;
   out_820181129074544695[15] = 0;
   out_820181129074544695[16] = 0;
   out_820181129074544695[17] = 0;
   out_820181129074544695[18] = 0;
   out_820181129074544695[19] = 0;
   out_820181129074544695[20] = 0;
   out_820181129074544695[21] = 0;
   out_820181129074544695[22] = 0;
   out_820181129074544695[23] = 0;
   out_820181129074544695[24] = 0;
   out_820181129074544695[25] = 1;
   out_820181129074544695[26] = 0;
   out_820181129074544695[27] = 0;
   out_820181129074544695[28] = 0;
   out_820181129074544695[29] = 0;
   out_820181129074544695[30] = 0;
   out_820181129074544695[31] = 0;
   out_820181129074544695[32] = 0;
   out_820181129074544695[33] = 0;
   out_820181129074544695[34] = 0;
   out_820181129074544695[35] = 0;
   out_820181129074544695[36] = 0;
   out_820181129074544695[37] = 0;
   out_820181129074544695[38] = 0;
   out_820181129074544695[39] = 0;
   out_820181129074544695[40] = 0;
   out_820181129074544695[41] = 0;
   out_820181129074544695[42] = 0;
   out_820181129074544695[43] = 0;
   out_820181129074544695[44] = 1;
   out_820181129074544695[45] = 0;
   out_820181129074544695[46] = 0;
   out_820181129074544695[47] = 0;
   out_820181129074544695[48] = 0;
   out_820181129074544695[49] = 0;
   out_820181129074544695[50] = 0;
   out_820181129074544695[51] = 0;
   out_820181129074544695[52] = 0;
   out_820181129074544695[53] = 0;
}
#include <eigen3/Eigen/Dense>
#include <iostream>

typedef Eigen::Matrix<double, DIM, DIM, Eigen::RowMajor> DDM;
typedef Eigen::Matrix<double, EDIM, EDIM, Eigen::RowMajor> EEM;
typedef Eigen::Matrix<double, DIM, EDIM, Eigen::RowMajor> DEM;

void predict(double *in_x, double *in_P, double *in_Q, double dt) {
  typedef Eigen::Matrix<double, MEDIM, MEDIM, Eigen::RowMajor> RRM;

  double nx[DIM] = {0};
  double in_F[EDIM*EDIM] = {0};

  // functions from sympy
  f_fun(in_x, dt, nx);
  F_fun(in_x, dt, in_F);


  EEM F(in_F);
  EEM P(in_P);
  EEM Q(in_Q);

  RRM F_main = F.topLeftCorner(MEDIM, MEDIM);
  P.topLeftCorner(MEDIM, MEDIM) = (F_main * P.topLeftCorner(MEDIM, MEDIM)) * F_main.transpose();
  P.topRightCorner(MEDIM, EDIM - MEDIM) = F_main * P.topRightCorner(MEDIM, EDIM - MEDIM);
  P.bottomLeftCorner(EDIM - MEDIM, MEDIM) = P.bottomLeftCorner(EDIM - MEDIM, MEDIM) * F_main.transpose();

  P = P + dt*Q;

  // copy out state
  memcpy(in_x, nx, DIM * sizeof(double));
  memcpy(in_P, P.data(), EDIM * EDIM * sizeof(double));
}

// note: extra_args dim only correct when null space projecting
// otherwise 1
template <int ZDIM, int EADIM, bool MAHA_TEST>
void update(double *in_x, double *in_P, Hfun h_fun, Hfun H_fun, Hfun Hea_fun, double *in_z, double *in_R, double *in_ea, double MAHA_THRESHOLD) {
  typedef Eigen::Matrix<double, ZDIM, ZDIM, Eigen::RowMajor> ZZM;
  typedef Eigen::Matrix<double, ZDIM, DIM, Eigen::RowMajor> ZDM;
  typedef Eigen::Matrix<double, Eigen::Dynamic, EDIM, Eigen::RowMajor> XEM;
  //typedef Eigen::Matrix<double, EDIM, ZDIM, Eigen::RowMajor> EZM;
  typedef Eigen::Matrix<double, Eigen::Dynamic, 1> X1M;
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> XXM;

  double in_hx[ZDIM] = {0};
  double in_H[ZDIM * DIM] = {0};
  double in_H_mod[EDIM * DIM] = {0};
  double delta_x[EDIM] = {0};
  double x_new[DIM] = {0};


  // state x, P
  Eigen::Matrix<double, ZDIM, 1> z(in_z);
  EEM P(in_P);
  ZZM pre_R(in_R);

  // functions from sympy
  h_fun(in_x, in_ea, in_hx);
  H_fun(in_x, in_ea, in_H);
  ZDM pre_H(in_H);

  // get y (y = z - hx)
  Eigen::Matrix<double, ZDIM, 1> pre_y(in_hx); pre_y = z - pre_y;
  X1M y; XXM H; XXM R;
  if (Hea_fun){
    typedef Eigen::Matrix<double, ZDIM, EADIM, Eigen::RowMajor> ZAM;
    double in_Hea[ZDIM * EADIM] = {0};
    Hea_fun(in_x, in_ea, in_Hea);
    ZAM Hea(in_Hea);
    XXM A = Hea.transpose().fullPivLu().kernel();


    y = A.transpose() * pre_y;
    H = A.transpose() * pre_H;
    R = A.transpose() * pre_R * A;
  } else {
    y = pre_y;
    H = pre_H;
    R = pre_R;
  }
  // get modified H
  H_mod_fun(in_x, in_H_mod);
  DEM H_mod(in_H_mod);
  XEM H_err = H * H_mod;

  // Do mahalobis distance test
  if (MAHA_TEST){
    XXM a = (H_err * P * H_err.transpose() + R).inverse();
    double maha_dist = y.transpose() * a * y;
    if (maha_dist > MAHA_THRESHOLD){
      R = 1.0e16 * R;
    }
  }

  // Outlier resilient weighting
  double weight = 1;//(1.5)/(1 + y.squaredNorm()/R.sum());

  // kalman gains and I_KH
  XXM S = ((H_err * P) * H_err.transpose()) + R/weight;
  XEM KT = S.fullPivLu().solve(H_err * P.transpose());
  //EZM K = KT.transpose(); TODO: WHY DOES THIS NOT COMPILE?
  //EZM K = S.fullPivLu().solve(H_err * P.transpose()).transpose();
  //std::cout << "Here is the matrix rot:\n" << K << std::endl;
  EEM I_KH = Eigen::Matrix<double, EDIM, EDIM>::Identity() - (KT.transpose() * H_err);

  // update state by injecting dx
  Eigen::Matrix<double, EDIM, 1> dx(delta_x);
  dx  = (KT.transpose() * y);
  memcpy(delta_x, dx.data(), EDIM * sizeof(double));
  err_fun(in_x, delta_x, x_new);
  Eigen::Matrix<double, DIM, 1> x(x_new);

  // update cov
  P = ((I_KH * P) * I_KH.transpose()) + ((KT.transpose() * R) * KT);

  // copy out state
  memcpy(in_x, x.data(), DIM * sizeof(double));
  memcpy(in_P, P.data(), EDIM * EDIM * sizeof(double));
  memcpy(in_z, y.data(), y.rows() * sizeof(double));
}




}
extern "C" {

void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<3, 3, 0>(in_x, in_P, h_4, H_4, NULL, in_z, in_R, in_ea, MAHA_THRESH_4);
}
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<3, 3, 0>(in_x, in_P, h_10, H_10, NULL, in_z, in_R, in_ea, MAHA_THRESH_10);
}
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<3, 3, 0>(in_x, in_P, h_13, H_13, NULL, in_z, in_R, in_ea, MAHA_THRESH_13);
}
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<3, 3, 0>(in_x, in_P, h_14, H_14, NULL, in_z, in_R, in_ea, MAHA_THRESH_14);
}
void pose_err_fun(double *nom_x, double *delta_x, double *out_6757027054840770867) {
  err_fun(nom_x, delta_x, out_6757027054840770867);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_3041472167094278981) {
  inv_err_fun(nom_x, true_x, out_3041472167094278981);
}
void pose_H_mod_fun(double *state, double *out_2705022079487018887) {
  H_mod_fun(state, out_2705022079487018887);
}
void pose_f_fun(double *state, double dt, double *out_5934707486323140053) {
  f_fun(state,  dt, out_5934707486323140053);
}
void pose_F_fun(double *state, double dt, double *out_6052904124511391966) {
  F_fun(state,  dt, out_6052904124511391966);
}
void pose_h_4(double *state, double *unused, double *out_2598800994798863224) {
  h_4(state, unused, out_2598800994798863224);
}
void pose_H_4(double *state, double *unused, double *out_3143059727264939834) {
  H_4(state, unused, out_3143059727264939834);
}
void pose_h_10(double *state, double *unused, double *out_1573633236380714031) {
  h_10(state, unused, out_1573633236380714031);
}
void pose_H_10(double *state, double *unused, double *out_7106231390164887414) {
  H_10(state, unused, out_7106231390164887414);
}
void pose_h_13(double *state, double *unused, double *out_7607426681960106713) {
  h_13(state, unused, out_7607426681960106713);
}
void pose_H_13(double *state, double *unused, double *out_69214098067392967) {
  H_13(state, unused, out_69214098067392967);
}
void pose_h_14(double *state, double *unused, double *out_5439578415485324028) {
  h_14(state, unused, out_5439578415485324028);
}
void pose_H_14(double *state, double *unused, double *out_820181129074544695) {
  H_14(state, unused, out_820181129074544695);
}
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt) {
  predict(in_x, in_P, in_Q, dt);
}
}

const EKF pose = {
  .name = "pose",
  .kinds = { 4, 10, 13, 14 },
  .feature_kinds = {  },
  .f_fun = pose_f_fun,
  .F_fun = pose_F_fun,
  .err_fun = pose_err_fun,
  .inv_err_fun = pose_inv_err_fun,
  .H_mod_fun = pose_H_mod_fun,
  .predict = pose_predict,
  .hs = {
    { 4, pose_h_4 },
    { 10, pose_h_10 },
    { 13, pose_h_13 },
    { 14, pose_h_14 },
  },
  .Hs = {
    { 4, pose_H_4 },
    { 10, pose_H_10 },
    { 13, pose_H_13 },
    { 14, pose_H_14 },
  },
  .updates = {
    { 4, pose_update_4 },
    { 10, pose_update_10 },
    { 13, pose_update_13 },
    { 14, pose_update_14 },
  },
  .Hes = {
  },
  .sets = {
  },
  .extra_routines = {
  },
};

ekf_lib_init(pose)
