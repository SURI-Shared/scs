#include "glbopts.h"
#include "minunit.h"
#include "problem_utils.h"
#include "scs.h"
#include "util.h"

scs_float P_x[] = {
    10.74980, 1.07498,  41.62331, 0.11891,  5.80636,  0.58064,  3.13259,
    0.31326,  41.62331, 41.62331, 41.62331, 41.62331, 41.62331, 41.62331,
    0.11891,  11.49095, 1.14910,  3.13259,  0.31326,  41.62331, 41.62331,
    41.62331, 41.62331, 41.62331, 41.62331, 0.11891,  11.49095, 1.14910,
    3.13259,  0.31326,  41.62331, 41.62331, 41.62331, 41.62331, 41.62331,
    41.62331, 0.11891,  11.49095, 1.14910,  50.00000, 41.62331, 41.62331,
    41.62331, 41.62331, 10.40583, 10.74980, 10.74980, 1.07498,  1.07498,
    50.00000, 3.13259,  3.13259,  0.31326,  0.31326,  50.00000, 41.62331,
    41.62331, 50.00000, 41.62331, 41.62331, 41.62331, 25.00000, 5.80636,
    0.58064,  41.62331, 41.62331, 41.62331, 41.62331, 41.62331, 41.62331,
    50.00000, 3.13259,  3.13259,  0.31326,  0.31326,  50.00000, 41.62331,
    41.62331, 50.00000, 41.62331, 41.62331, 41.62331, 25.00000, 5.80636,
    0.58064,  41.62331, 41.62331, 41.62331, 41.62331, 41.62331, 41.62331,
    50.00000, 3.13259,  3.13259,  0.31326,  0.31326,  50.00000, 41.62331,
    41.62331, 50.00000, 41.62331, 41.62331, 41.62331, 25.00000, 5.80636,
    0.58064,  41.62331, 41.62331, 41.62331, 41.62331, 41.62331, 41.62331,
    41.62331, 0.19531,  10.40583, 10.40583, 0.19531,  41.62331, 41.62331,
    0.19531,  41.62331, 41.62331, 0.19531,  41.62331, 41.62331, 0.19531,
    41.62331, 41.62331, 0.19531,  41.62331, 41.62331, 0.19531,  41.62331,
    41.62331};
scs_int P_i[] = {
    1,   2,   4,   5,   6,   7,   9,   10,  12,  14,  16,  18,  20,  22,  23,
    24,  25,  27,  28,  30,  32,  34,  36,  38,  40,  41,  42,  43,  45,  46,
    48,  50,  52,  54,  56,  58,  59,  60,  61,  62,  63,  65,  67,  69,  71,
    1,   73,  2,   74,  75,  9,   76,  10,  77,  78,  12,  79,  80,  14,  81,
    83,  84,  85,  86,  88,  90,  92,  94,  20,  96,  97,  27,  98,  28,  99,
    100, 30,  101, 102, 32,  103, 105, 106, 107, 108, 110, 112, 114, 116, 38,
    118, 119, 45,  120, 46,  121, 122, 48,  123, 124, 50,  125, 127, 128, 129,
    130, 132, 134, 136, 63,  138, 56,  140, 141, 71,  142, 143, 16,  144, 145,
    22,  146, 147, 34,  148, 149, 40,  150, 151, 52,  152, 153, 58,  154};
scs_int P_p[] = {
    0,   0,   1,   2,   2,   3,   4,   5,   6,   6,   7,   8,   8,   9,   9,
    10,  10,  11,  11,  12,  12,  13,  13,  14,  15,  16,  17,  17,  18,  19,
    19,  20,  20,  21,  21,  22,  22,  23,  23,  24,  24,  25,  26,  27,  28,
    28,  29,  30,  30,  31,  31,  32,  32,  33,  33,  34,  34,  35,  35,  36,
    37,  38,  39,  40,  41,  41,  42,  42,  43,  43,  44,  44,  45,  45,  47,
    49,  50,  52,  54,  55,  57,  58,  60,  60,  61,  62,  63,  64,  64,  65,
    65,  66,  66,  67,  67,  68,  68,  70,  71,  73,  75,  76,  78,  79,  81,
    81,  82,  83,  84,  85,  85,  86,  86,  87,  87,  88,  88,  89,  89,  91,
    92,  94,  96,  97,  99,  100, 102, 102, 103, 104, 105, 106, 106, 107, 107,
    108, 108, 109, 109, 111, 111, 113, 114, 116, 117, 119, 120, 122, 123, 125,
    126, 128, 129, 131, 132, 134};
scs_float A_x[] = {
    1.00000,  1.00000,  1.00000,  -1.00000, 1.00000,  1.00000,  1.00000,
    1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  -1.00000, 1.00000,
    1.00000,  1.00000,  1.00000,  1.00000,  -1.00000, 1.00000,  1.00000,
    1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  -1.00000,
    1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  1.00000,
    1.00000,  -1.00000, 1.00000,  1.00000,  1.00000,  1.00000,  1.00000,
    -1.00000, 1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  -1.00000,
    1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  -1.00000, 1.00000,
    1.00000,  1.00000,  1.00000,  1.00000,  -1.00000, 1.00000,  1.00000,
    1.00000,  1.00000,  1.00000,  -1.00000, 1.00000,  1.00000,  1.00000,
    1.00000,  1.00000,  -1.00000, 1.00000,  1.00000,  1.00000,  1.00000,
    1.00000,  1.00000,  1.00000,  1.00000,  -1.00000, 1.00000,  1.00000,
    1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  -1.00000,
    1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  -1.00000, 1.00000,
    1.00000,  1.00000,  1.00000,  1.00000,  -1.00000, 1.00000,  1.00000,
    1.00000,  1.00000,  1.00000,  -1.00000, 1.00000,  1.00000,  1.00000,
    1.00000,  1.00000,  -1.00000, 1.00000,  1.00000,  1.00000,  1.00000,
    1.00000,  -1.00000, 1.00000,  1.00000,  1.00000,  1.00000,  1.00000,
    -1.00000, 1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  1.00000,
    1.00000,  1.00000,  -1.00000, 1.00000,  1.00000,  1.00000,  1.00000,
    1.00000,  1.00000,  1.00000,  1.00000,  -1.00000, 1.00000,  1.00000,
    1.00000,  1.00000,  1.00000,  -1.00000, 1.00000,  1.00000,  1.00000,
    1.00000,  1.00000,  -1.00000, 1.00000,  1.00000,  1.00000,  1.00000,
    1.00000,  -1.00000, 1.00000,  1.00000,  1.00000,  1.00000,  1.00000,
    -1.00000, 1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  -1.00000,
    1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  -1.00000, 1.00000,
    1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  1.00000,
    -1.00000, 1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  -1.00000,
    1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  -1.00000, 1.00000,
    1.00000,  1.00000,  1.00000,  1.00000,  -1.00000, 1.00000,  1.00000,
    1.00000,  1.00000,  1.00000,  -1.00000, 1.00000,  1.00000,  1.00000,
    1.00000,  1.00000,  -1.00000, 1.00000,  1.00000,  1.00000,  1.00000,
    1.00000,  1.00000,  1.00000,  1.00000,  -1.00000, 1.00000,  1.00000,
    1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  -1.00000,
    1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  -1.00000, 1.00000,
    1.00000,  1.00000,  1.00000,  1.00000,  -1.00000, 1.00000,  1.00000,
    1.00000,  1.00000,  1.00000,  -1.00000, 1.00000,  1.00000,  1.00000,
    1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  -1.00000, 1.00000,
    1.00000,  1.00000,  1.00000,  1.00000,  -1.00000, 1.00000,  1.00000,
    1.00000,  1.00000,  1.00000,  -1.00000, 1.00000,  1.00000,  1.00000,
    1.00000,  1.00000,  -1.00000, 1.00000,  1.00000,  1.00000,  1.00000,
    1.00000,  -1.00000, 1.00000,  1.00000,  1.00000,  1.00000,  1.00000,
    -1.00000, 1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  1.00000,
    1.00000,  1.00000,  -1.00000, 1.00000,  1.00000,  1.00000,  1.00000,
    1.00000,  -1.00000, 1.00000,  1.00000,  1.00000,  1.00000,  1.00000,
    -1.00000, 1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  -1.00000,
    1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  1.00000,
    1.00000,  -1.00000, 1.00000,  1.00000,  1.00000,  1.00000,  1.00000,
    -1.00000, 1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  -1.00000,
    1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  -1.00000, 1.00000,
    1.00000,  1.00000,  1.00000,  1.00000,  -1.00000, 1.00000,  1.00000,
    1.00000,  1.00000,  1.00000,  -1.00000, 1.00000,  1.00000,  1.00000,
    1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  -1.00000, 1.00000,
    1.00000,  1.00000,  1.00000,  1.00000,  -1.00000, 1.00000,  1.00000,
    1.00000,  1.00000,  1.00000,  -1.00000, 1.00000,  1.00000,  1.00000,
    1.00000,  1.00000,  -1.00000, 1.00000,  1.00000,  1.00000,  1.00000,
    1.00000,  1.00000,  1.00000,  1.00000,  -1.00000, 1.00000,  1.00000,
    1.00000,  1.00000,  1.00000,  -1.00000, 1.00000,  1.00000,  1.00000,
    1.00000,  1.00000,  -1.00000, 1.00000,  1.00000,  1.00000,  1.00000,
    1.00000,  -1.00000, 1.00000,  1.00000,  1.00000,  1.00000,  1.00000,
    -1.00000, 1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  -1.00000,
    1.00000,  1.00000,  1.00000,  1.00000,  1.00000,  -1.00000, 1.00000,
    1.00000,  1.00000,  1.00000,  1.00000,  -1.00000, 1.00000,  1.00000,
    1.00000,  1.00000,  1.00000,  -1.00000, 1.00000,  1.00000,  1.00000,
    1.00000,  1.00000,  -1.00000, 1.00000,  1.00000,  1.00000,  1.00000,
    1.00000,  -1.00000, 1.00000,  1.00000,  1.00000,  1.00000,  1.00000,
    -1.00000, 1.00000,  1.00000};
scs_int A_i[] = {
    0,   3,   34,  96,  96,  197, 96,  166, 177, 1,   3,   34,  97,  97,  199,
    2,   3,   34,  98,  98,  200, 98,  167, 179, 4,   9,   34,  99,  99,  201,
    99,  168, 180, 5,   9,   34,  100, 100, 203, 6,   9,   34,  101, 101, 205,
    7,   9,   34,  102, 102, 207, 8,   9,   34,  103, 103, 209, 10,  13,  34,
    104, 104, 216, 11,  13,  34,  105, 105, 218, 12,  13,  34,  106, 106, 220,
    106, 170, 183, 14,  19,  34,  107, 107, 221, 107, 171, 184, 15,  19,  34,
    108, 108, 223, 16,  19,  34,  109, 109, 225, 17,  19,  34,  110, 110, 227,
    18,  19,  34,  111, 111, 229, 20,  23,  34,  112, 112, 236, 21,  23,  34,
    113, 113, 238, 22,  23,  34,  114, 114, 240, 114, 173, 187, 24,  29,  34,
    115, 115, 241, 115, 174, 188, 25,  29,  34,  116, 116, 243, 26,  29,  34,
    117, 117, 245, 27,  29,  34,  118, 118, 247, 28,  29,  34,  119, 119, 249,
    30,  33,  34,  120, 120, 257, 31,  33,  34,  121, 121, 259, 32,  33,  34,
    122, 122, 261, 122, 176, 191, 35,  36,  37,  123, 123, 255, 38,  43,  80,
    124, 124, 192, 39,  43,  80,  125, 125, 193, 40,  43,  80,  126, 126, 194,
    41,  43,  80,  127, 127, 195, 42,  43,  80,  128, 128, 198, 128, 166, 178,
    44,  49,  80,  129, 129, 202, 129, 168, 181, 45,  49,  80,  130, 130, 204,
    46,  49,  80,  131, 131, 206, 47,  49,  80,  132, 132, 210, 48,  49,  80,
    133, 133, 211, 133, 169, 182, 50,  55,  80,  134, 134, 212, 51,  55,  80,
    135, 135, 213, 52,  55,  80,  136, 136, 214, 53,  55,  80,  137, 137, 215,
    54,  55,  80,  138, 138, 217, 56,  61,  80,  139, 139, 222, 139, 171, 185,
    57,  61,  80,  140, 140, 224, 58,  61,  80,  141, 141, 226, 59,  61,  80,
    142, 142, 230, 60,  61,  80,  143, 143, 231, 143, 172, 186, 62,  67,  80,
    144, 144, 232, 63,  67,  80,  145, 145, 233, 64,  67,  80,  146, 146, 234,
    65,  67,  80,  147, 147, 235, 66,  67,  80,  148, 148, 237, 68,  73,  80,
    149, 149, 242, 149, 174, 189, 69,  73,  80,  150, 150, 244, 70,  73,  80,
    151, 151, 246, 71,  73,  80,  152, 152, 250, 72,  73,  80,  153, 153, 251,
    153, 175, 190, 74,  79,  80,  154, 154, 252, 75,  79,  80,  155, 155, 253,
    76,  79,  80,  156, 156, 254, 77,  79,  80,  157, 157, 256, 78,  79,  80,
    158, 158, 258, 81,  82,  95,  159, 159, 196, 83,  84,  95,  160, 160, 208,
    85,  86,  95,  161, 161, 219, 87,  88,  95,  162, 162, 228, 89,  90,  95,
    163, 163, 239, 91,  92,  95,  164, 164, 248, 93,  94,  95,  165, 165, 260};
scs_int A_p[] = {
    0,   4,   6,   9,   13,  15,  19,  21,  24,  28,  30,  33,  37,  39,  43,
    45,  49,  51,  55,  57,  61,  63,  67,  69,  73,  75,  78,  82,  84,  87,
    91,  93,  97,  99,  103, 105, 109, 111, 115, 117, 121, 123, 127, 129, 132,
    136, 138, 141, 145, 147, 151, 153, 157, 159, 163, 165, 169, 171, 175, 177,
    181, 183, 186, 190, 192, 196, 198, 202, 204, 208, 210, 214, 216, 220, 222,
    225, 229, 231, 234, 238, 240, 244, 246, 250, 252, 256, 258, 261, 265, 267,
    271, 273, 277, 279, 283, 285, 289, 291, 295, 297, 300, 304, 306, 310, 312,
    316, 318, 322, 324, 327, 331, 333, 337, 339, 343, 345, 349, 351, 355, 357,
    361, 363, 366, 370, 372, 376, 378, 382, 384, 388, 390, 393, 397, 399, 403,
    405, 409, 411, 415, 417, 421, 423, 427, 429, 433, 435, 439, 441, 445, 447,
    451, 453, 457, 459, 463, 465};
scs_int n = 155;
scs_int m = 262;
scs_float l[] = {
    30.00000, 30.00000,  59.00000,   119.00000, 30.00000,  30.00000,
    30.00000, 30.00000,  30.00000,   150.00000, 30.00000,  30.00000,
    59.00000, 119.00000, 30.00000,   30.00000,  30.00000,  30.00000,
    30.00000, 150.00000, 30.00000,   30.00000,  59.00000,  119.00000,
    30.00000, 30.00000,  30.00000,   30.00000,  30.00000,  150.00000,
    30.00000, 30.00000,  59.00000,   119.00000, 926.00000, 31.00000,
    31.00000, 31.00000,  30.00000,   30.00000,  30.00000,  30.00000,
    30.00000, 150.00000, 31.00000,   31.00000,  31.00000,  30.00000,
    28.00000, 151.00000, 30.00000,   30.00000,  30.00000,  30.00000,
    30.00000, 150.00000, 31.00000,   31.00000,  31.00000,  30.00000,
    28.00000, 151.00000, 30.00000,   30.00000,  30.00000,  30.00000,
    30.00000, 150.00000, 31.00000,   31.00000,  31.00000,  30.00000,
    28.00000, 151.00000, 30.00000,   30.00000,  30.00000,  30.00000,
    30.00000, 150.00000, 1053.00000, 16.00000,  16.00000,  16.00000,
    16.00000, 16.00000,  16.00000,   16.00000,  16.00000,  16.00000,
    16.00000, 16.00000,  16.00000,   16.00000,  16.00000,  112.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   0.00000,   0.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   0.00000,   0.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   0.00000,   0.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   0.00000,   0.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   0.00000,   0.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   0.00000,   0.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   0.00000,   0.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   0.00000,   0.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   0.00000,   0.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   0.00000,   0.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   0.00000,   0.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   0.00000,   0.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   0.00000,   0.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   0.00000,   0.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   0.00000,   0.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   0.00000,   0.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   0.00000,   0.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   0.00000,   0.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   0.00000,   0.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   0.00000,   0.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   0.00000,   0.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   0.00000,   0.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   0.00000,   0.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   0.00000,   0.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   0.00000,   0.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   0.00000,   0.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   0.00000,   0.00000,
    0.00000,  0.00000,   0.00000,    0.00000};
scs_float u[] = {
    30.00000, 30.00000,  59.00000,   119.00000, 30.00000,  30.00000,
    30.00000, 30.00000,  30.00000,   150.00000, 30.00000,  30.00000,
    59.00000, 119.00000, 30.00000,   30.00000,  30.00000,  30.00000,
    30.00000, 150.00000, 30.00000,   30.00000,  59.00000,  119.00000,
    30.00000, 30.00000,  30.00000,   30.00000,  30.00000,  150.00000,
    30.00000, 30.00000,  59.00000,   119.00000, 926.00000, 31.00000,
    31.00000, 31.00000,  30.00000,   30.00000,  30.00000,  30.00000,
    30.00000, 150.00000, 31.00000,   31.00000,  31.00000,  30.00000,
    28.00000, 151.00000, 30.00000,   30.00000,  30.00000,  30.00000,
    30.00000, 150.00000, 31.00000,   31.00000,  31.00000,  30.00000,
    28.00000, 151.00000, 30.00000,   30.00000,  30.00000,  30.00000,
    30.00000, 150.00000, 31.00000,   31.00000,  31.00000,  30.00000,
    28.00000, 151.00000, 30.00000,   30.00000,  30.00000,  30.00000,
    30.00000, 150.00000, 1053.00000, 36.00000,  17.00000,  36.00000,
    17.00000, 36.00000,  17.00000,   36.00000,  17.00000,  36.00000,
    17.00000, 36.00000,  17.00000,   36.00000,  17.00000,  112.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   0.00000,   0.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   0.00000,   0.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   0.00000,   0.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   0.00000,   0.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   0.00000,   0.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   0.00000,   0.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   0.00000,   0.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   0.00000,   0.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   0.00000,   0.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   0.00000,   0.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   0.00000,   0.00000,
    0.00000,  0.00000,   0.00000,    0.00000,   30.00000,  28.00000,
    54.00000, 52.00000,  28.00000,   54.00000,  52.00000,  28.00000,
    52.00000, 52.00000,  28.00000,   96.00000,  96.00000,  96.00000,
    96.00000, 96.00000,  96.00000,   96.00000,  96.00000,  96.00000,
    96.00000, 96.00000,  96.00000,   96.00000,  96.00000,  96.00000,
    96.00000, 96.00000,  96.00000,   96.00000,  96.00000,  96.00000,
    96.00000, 96.00000,  96.00000,   96.00000,  96.00000,  96.00000,
    96.00000, 96.00000,  96.00000,   96.00000,  96.00000,  96.00000,
    96.00000, 96.00000,  96.00000,   96.00000,  96.00000,  96.00000,
    96.00000, 96.00000,  96.00000,   96.00000,  96.00000,  96.00000,
    96.00000, 96.00000,  96.00000,   96.00000,  96.00000,  96.00000,
    96.00000, 96.00000,  96.00000,   96.00000,  96.00000,  96.00000,
    96.00000, 96.00000,  96.00000,   96.00000,  96.00000,  96.00000,
    96.00000, 96.00000,  96.00000,   96.00000,  96.00000,  96.00000,
    96.00000, 96.00000,  96.00000,   96.00000,  96.00000,  96.00000,
    96.00000, 96.00000,  96.00000,   96.00000,  96.00000,  96.00000,
    96.00000, 96.00000,  96.00000,   96.00000};
scs_float q[] = {0.00000,     0.00000,    -35.02643,   0.00000,     0.00000,
                 -3.56718,    0.00000,    -17.41907,   0.00000,     0.00000,
                 -17.64691,   0.00000,    0.00000,     0.00000,     0.00000,
                 0.00000,     0.00000,    0.00000,     0.00000,     0.00000,
                 0.00000,     0.00000,    0.00000,     -3.56718,    0.00000,
                 -32.17466,   0.00000,    0.00000,     -17.64691,   0.00000,
                 0.00000,     0.00000,    0.00000,     0.00000,     0.00000,
                 0.00000,     0.00000,    0.00000,     0.00000,     0.00000,
                 0.00000,     -3.56718,   0.00000,     -32.17466,   0.00000,
                 0.00000,     -17.07260,  0.00000,     0.00000,     0.00000,
                 0.00000,     0.00000,    0.00000,     0.00000,     0.00000,
                 0.00000,     0.00000,    0.00000,     0.00000,     -3.56718,
                 0.00000,     -32.17466,  -1600.00000, 0.00000,     0.00000,
                 0.00000,     0.00000,    0.00000,     0.00000,     0.00000,
                 0.00000,     0.00000,    0.00000,     0.00000,     -35.02643,
                 -1500.00000, 0.00000,    -17.64691,   -1500.00000, 0.00000,
                 -1500.00000, 0.00000,    0.00000,     0.00000,     -750.00000,
                 0.00000,     -31.69304,  0.00000,     0.00000,     0.00000,
                 0.00000,     0.00000,    0.00000,     0.00000,     0.00000,
                 0.00000,     0.00000,    -1500.00000, 0.00000,     -17.64691,
                 -1500.00000, 0.00000,    -1500.00000, 0.00000,     0.00000,
                 0.00000,     -750.00000, 0.00000,     -31.69304,   0.00000,
                 0.00000,     0.00000,    0.00000,     0.00000,     0.00000,
                 0.00000,     0.00000,    0.00000,     0.00000,     -1500.00000,
                 0.00000,     -17.07260,  -1500.00000, 0.00000,     -1500.00000,
                 0.00000,     0.00000,    0.00000,     -750.00000,  0.00000,
                 -31.69304,   0.00000,    0.00000,     0.00000,     0.00000,
                 0.00000,     0.00000,    0.00000,     0.00000,     0.00000,
                 0.00000,     -3.90625,   0.00000,     -3.90625,    0.00000,
                 -3.90625,    0.00000,    -3.90625,    0.00000,     -3.90625,
                 0.00000,     -3.90625,   0.00000,     -3.90625,    0.00000};
scs_int A_nnz = 465;

static const char *small_qp(void) {
  ScsCone *k = (ScsCone *)scs_calloc(1, sizeof(ScsCone));
  ScsData *d = (ScsData *)scs_calloc(1, sizeof(ScsData));
  ScsSettings *stgs = (ScsSettings *)scs_calloc(1, sizeof(ScsSettings));
  ScsSolution *sol = (ScsSolution *)scs_calloc(1, sizeof(ScsSolution));
  ScsInfo info = {0};
  scs_int success, exitflag;
  scs_int j;
  const char *fail;

  d->m = m + 1; /* t var in box cone */
  d->n = n;
  d->b = (scs_float *)scs_calloc(m + 1, sizeof(scs_float));
  d->b[0] = 1;  /* t var in box cone */
  d->c = q;

  d->A = (ScsMatrix *)scs_calloc(1, sizeof(ScsMatrix));
  d->A->m = m + 1; /* t var in box cone */
  d->A->n = n;
  d->A->x = A_x;
  d->A->i = A_i;
  d->A->p = A_p;

  /* Ax + s = b, s \in box, need to negate A */
  for (j = 0; j < A_nnz; ++j) {
    d->A->x[j] *= -1;
  }
  /* need to add row of all zeros to top of A */
  for (j = 0; j < A_nnz; ++j) {
    d->A->i[j] += 1;
  }

  d->P = (ScsMatrix *)scs_calloc(1, sizeof(ScsMatrix));
  d->P->m = n;
  d->P->n = n;
  d->P->x = P_x;
  d->P->i = P_i;
  d->P->p = P_p;

  k->bu = u;
  k->bl = l;
  k->bsize = m + 1; /* t var in box cone */

  SCS(set_default_settings)(stgs);
  stgs->eps_abs = 1e-6;
  stgs->eps_rel = 1e-6;
  stgs->eps_infeas = 1e-10;

  exitflag = scs(d, k, stgs, sol, &info);
  success = exitflag == SCS_SOLVED;

  mu_assert("small_qp: SCS failed to produce outputflag SCS_SOLVED", success);
  fail = verify_solution_correct(d, k, stgs, &info, sol, exitflag);

  scs_free(d->A);
  scs_free(d->P);
  scs_free(d->b);
  scs_free(k);
  scs_free(d);
  scs_free(stgs);
  scs_free(sol->x);
  scs_free(sol->y);
  scs_free(sol->s);
  scs_free(sol);
  return fail;
}