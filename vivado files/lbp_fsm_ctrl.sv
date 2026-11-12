// ============================================================================
// lbp_fsm_ctrl.sv
// Frame sequencer. Owns the phase order so the datapath blocks never collide:
//
//   IDLE -> CLEAR -> STREAM -> FLUSH -> DRAIN -> CLASSIFY -> DONE -> IDLE
//
//   CLEAR    : histogram sweep-clear (4096 clk); engine counters reset
//   STREAM   : accept 65536 pixels (pix_valid & pix_ready), each = one engine step
//   FLUSH    : 257 engine steps with pix = 0 to push out the last line of centres
//   DRAIN    : wait for the engine's final code (code_last) + RMW write-back
//   CLASSIFY : pulse clf_start; wait for clf_done
//   DONE     : pulse done, return to IDLE
//
// Throttle: an engine step is never issued in two consecutive clocks, so
// code_valid pulses are >= 2 clk apart (histogram RMW requirement).
// ============================================================================
module lbp_fsm_ctrl (
    input  logic clk,
    input  logic rst_n,
    input  logic start,

    // pixel handshake
    input  logic pix_valid,
    output logic pix_ready,

    // to lbp_3x3_engine
    output logic frame_start,
    output logic eng_en,
    output logic eng_flush,        // select zero pixel instead of pix_data
    input  logic code_last,

    // to histogram_acc_q14
    output logic hist_clr_start,
    input  logic hist_clr_busy,

    // to linear_svm_classifier
    output logic clf_start,
    input  logic clf_done,

    output logic busy,
    output logic done
);
    localparam int N_PIX   = 256*256;
    localparam int N_FLUSH = 257;

    typedef enum logic [2:0] {
        S_IDLE, S_CLEAR, S_STREAM, S_FLUSH, S_DRAIN, S_CLASSIFY, S_DONE
    } state_t;
    state_t state;

    logic [16:0] cnt;
    logic        step_d;                       // an engine step happened last clk
    logic        clr_issued;
    logic        drain_wait;

    wire can_step = !step_d;
    assign pix_ready = (state == S_STREAM) && can_step;
    wire   pix_xfer  = pix_valid && pix_ready;
    wire   flush_stp = (state == S_FLUSH) && can_step;

    assign eng_en    = pix_xfer || flush_stp;
    assign eng_flush = (state == S_FLUSH);
    assign busy      = (state != S_IDLE);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= S_IDLE;
            cnt            <= '0;
            step_d         <= 1'b0;
            clr_issued     <= 1'b0;
            drain_wait     <= 1'b0;
            frame_start    <= 1'b0;
            hist_clr_start <= 1'b0;
            clf_start      <= 1'b0;
            done           <= 1'b0;
        end else begin
            step_d         <= eng_en;
            frame_start    <= 1'b0;
            hist_clr_start <= 1'b0;
            clf_start      <= 1'b0;
            done           <= 1'b0;

            case (state)
                S_IDLE: if (start) begin
                    frame_start    <= 1'b1;
                    hist_clr_start <= 1'b1;
                    clr_issued     <= 1'b0;
                    cnt            <= '0;
                    state          <= S_CLEAR;
                end

                S_CLEAR: begin
                    // wait for the sweep to start, then to finish
                    if (hist_clr_busy) clr_issued <= 1'b1;
                    if (clr_issued && !hist_clr_busy) state <= S_STREAM;
                end

                S_STREAM: if (pix_xfer) begin
                    cnt <= cnt + 17'd1;
                    if (cnt == N_PIX-1) begin
                        cnt   <= '0;
                        state <= S_FLUSH;
                    end
                end

                S_FLUSH: if (flush_stp) begin
                    cnt <= cnt + 17'd1;
                    if (cnt == N_FLUSH-1) begin
                        cnt        <= '0;
                        drain_wait <= 1'b0;
                        state      <= S_DRAIN;
                    end
                end

                S_DRAIN: begin
                    // code_last is registered in the engine; one more clk for the
                    // histogram write-back of that last code before reading
                    if (code_last) drain_wait <= 1'b1;
                    if (drain_wait) begin
                        clf_start <= 1'b1;
                        state     <= S_CLASSIFY;
                    end
                end

                S_CLASSIFY: if (clf_done) state <= S_DONE;

                S_DONE: begin
                    done  <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
