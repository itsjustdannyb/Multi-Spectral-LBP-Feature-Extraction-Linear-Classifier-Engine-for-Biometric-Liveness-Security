// ============================================================================
// liveness_top.sv
// Biometric liveness engine: 256x256 gray frame in -> live/spoof decision out.
//
//   pix stream -> lbp_3x3_engine -> histogram_acc_q14 -> linear_svm_classifier -> live
//                       ^                  ^                     ^
//                       +------------ lbp_fsm_ctrl --------------+
//
// Usage: pulse start, then present 65536 pixels in raster order on
// pix_data/pix_valid (transfer when pix_valid && pix_ready). done pulses when
// score/live are valid; they hold until the next frame.
// ============================================================================
module liveness_top #(
    parameter WEIGHT_FILE = "svm_weights.mem",
    parameter int            W_FRAC      = 10,
    parameter signed [47:0]  BIAS        = 48'sd0
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,

    input  logic [7:0]  pix_data,
    input  logic        pix_valid,
    output logic        pix_ready,

    output logic        busy,
    output logic        done,
    output logic        live,            // 1 = live, 0 = spoof
    output logic signed [47:0] score,
    output logic        hist_overrun,

    // debug taps (for CSV dump / comparison with the Python model)
    output logic        dbg_lbp_valid,
    output logic [7:0]  dbg_lbp_code,
    output logic [3:0]  dbg_blk_idx
);
    // ctrl <-> engine
    logic       frame_start, eng_en, eng_flush, code_last;
    // engine -> histogram
    logic       code_valid;
    logic [7:0] lbp_code;
    logic [3:0] blk_idx;
    // ctrl <-> histogram
    logic       hist_clr_start, hist_clr_busy;
    // histogram <-> classifier
    logic        feat_rd_en, feat_valid;
    logic [11:0] feat_rd_addr;
    logic [14:0] feat_data;
    // ctrl <-> classifier
    logic       clf_start, clf_done;
    logic       clf_busy;

    lbp_fsm_ctrl u_ctrl (
        .clk, .rst_n, .start,
        .pix_valid, .pix_ready,
        .frame_start, .eng_en, .eng_flush, .code_last,
        .hist_clr_start, .hist_clr_busy,
        .clf_start, .clf_done,
        .busy, .done
    );

    lbp_3x3_engine u_lbp (
        .clk, .rst_n, .frame_start,
        .en      (eng_en),
        .pix_in  (eng_flush ? 8'd0 : pix_data),
        .code_valid, .lbp_code, .blk_idx, .code_last
    );

    histogram_acc_q14 u_hist (
        .clk, .rst_n,
        .clr_start (hist_clr_start), .clr_busy (hist_clr_busy),
        .code_valid, .blk_idx, .lbp_code, .overrun (hist_overrun),
        .rd_en (feat_rd_en), .rd_addr (feat_rd_addr),
        .rd_valid (feat_valid), .rd_data (feat_data)
    );

    linear_svm_classifier #(
        .WEIGHT_FILE (WEIGHT_FILE), .W_FRAC (W_FRAC), .BIAS (BIAS)
    ) u_svm (
        .clk, .rst_n, .start (clf_start),
        .feat_rd_en, .feat_rd_addr, .feat_valid, .feat_data,
        .busy (clf_busy), .done (clf_done), .score, .live
    );

    assign dbg_lbp_valid = code_valid;
    assign dbg_lbp_code  = lbp_code;
    assign dbg_blk_idx   = blk_idx;
endmodule
