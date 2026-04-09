#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_6757027054840770867);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_3041472167094278981);
void pose_H_mod_fun(double *state, double *out_2705022079487018887);
void pose_f_fun(double *state, double dt, double *out_5934707486323140053);
void pose_F_fun(double *state, double dt, double *out_6052904124511391966);
void pose_h_4(double *state, double *unused, double *out_2598800994798863224);
void pose_H_4(double *state, double *unused, double *out_3143059727264939834);
void pose_h_10(double *state, double *unused, double *out_1573633236380714031);
void pose_H_10(double *state, double *unused, double *out_7106231390164887414);
void pose_h_13(double *state, double *unused, double *out_7607426681960106713);
void pose_H_13(double *state, double *unused, double *out_69214098067392967);
void pose_h_14(double *state, double *unused, double *out_5439578415485324028);
void pose_H_14(double *state, double *unused, double *out_820181129074544695);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}