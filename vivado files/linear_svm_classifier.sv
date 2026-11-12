// ============================================================================
// linear_svm_classifier.sv
// Linear decision function over the 4096-element LBP feature vector:
//     score = sum_i  w[i] * x[i]  + bias        live = (score > 0)
// Same arithmetic as sklearn LogisticRegression / LinearSVC decision_function
// (coef_ -> weights, intercept_ -> BIAS), so either model can be exported.
//
// Number formats
//   x[i]  : unsigned Q0.14 (from histogram_acc_q14)
//   w[i]  : signed 16-bit, W_FRAC fractional bits (default Q5.10, range +-32)
//   score : signed ACC_W bits, scale 2^(14+W_FRAC)
//   BIAS  : round(intercept * 2^(14+W_FRAC))
// Weights are loaded from WEIGHT_FILE (4096 lines of 4 hex digits, two's
// complement) with $readmemh - generate it from the Python model.
//
// Operation: on start, scans feature addresses 0..4095 (one per clock),
// reading the histogram RAM and the weight ROM in lock-step (both have a
// 1-clock read latency), multiply-accumulates, then adds BIAS and pulses done.
// ============================================================================
module linear_svm_classifier #(
    parameter WEIGHT_FILE = "svm_weights.mem",
    parameter int                   W_FRAC      = 10,
    parameter signed [47:0]         BIAS        = 48'sd0
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,

    // feature fetch (to histogram_acc_q14)
    output logic        feat_rd_en,
    output logic [11:0] feat_rd_addr,
    input  logic        feat_valid,
    input  logic [14:0] feat_data,

    output logic        busy,
    output logic        done,        // 1-cycle pulse, score/live valid with it and after
    output logic signed [47:0] score,
    output logic        live
);
    localparam int ACC_W = 48;       // 15b x 16b product (31b) + 12b for 4096 terms = 43b

    logic signed [15:0] rom [0:4095];
    initial $readmemh(WEIGHT_FILE, rom);

    logic signed [15:0]    w_q;
    logic signed [ACC_W-1:0] acc;
    logic [1:0]            last_sh;

    typedef enum logic [0:0] {S_IDLE, S_RUN} state_t;
    state_t state;

    assign feat_rd_en = (state == S_RUN);
    assign busy       = (state == S_RUN) || (last_sh != 2'b00);

    wire signed [15:0] x_s = {1'b0, feat_data};            // Q0.14 as non-negative signed
    wire signed [31:0] prod = x_s * w_q;

    always_ff @(posedge clk) begin
        if (feat_rd_en) w_q <= rom[feat_rd_addr];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            feat_rd_addr <= '0;
            acc          <= '0;
            last_sh      <= '0;
            score        <= '0;
            live         <= 1'b0;
            done         <= 1'b0;
        end else begin
            done    <= 1'b0;
            last_sh <= {last_sh[0], 1'b0};

            case (state)
                S_IDLE: if (start) begin
                    state        <= S_RUN;
                    feat_rd_addr <= '0;
                    acc          <= '0;
                end
                S_RUN: begin
                    feat_rd_addr <= feat_rd_addr + 12'd1;
                    if (feat_rd_addr == 12'd4095) begin
                        state      <= S_IDLE;
                        last_sh[0] <= 1'b1;
                    end
                end
            endcase

            if (feat_valid) acc <= acc + ACC_W'(prod);

            // last_sh[0]: last address issued, last feature arrives this cycle
            // last_sh[1]: acc holds the full sum
            if (last_sh[1]) begin
                score <= acc + BIAS;
                live  <= ((acc + BIAS) > 0);
                done  <= 1'b1;
            end
        end
    end
endmodule
