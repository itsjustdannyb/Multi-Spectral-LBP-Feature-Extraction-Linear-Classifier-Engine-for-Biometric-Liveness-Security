// ============================================================================
// histogram_acc_q14.sv
// 16 blocks x 256 bins LBP histogram accumulator with Q14 normalised readout.
//
// Storage : 4096 x 13-bit counters, address = {blk_idx, lbp_code}.
// Update  : 2-cycle read-modify-write on a single-port synchronous RAM, so
//           code_valid pulses must be >= 2 clocks apart (lbp_fsm_ctrl
//           guarantees this). A violation sets overrun (sticky until clr_start).
// Readout : feature = count * 2^14 / 4096 = count << 2  -> unsigned Q0.14
//           (0..16384), the same "h / block.size" normalisation as the Python
//           model. Feature index {blk,bin} = blk*256 + bin, i.e. the layout of
//           np.concatenate(hists). rd_data is valid when rd_valid is high
//           (one clock after rd_en).
// Clear   : 4096-cycle sweep started by clr_start.
// Clear / accumulate / readout phases must not overlap (owned by lbp_fsm_ctrl).
// ============================================================================
module histogram_acc_q14 (
    input  logic        clk,
    input  logic        rst_n,

    // clear sweep
    input  logic        clr_start,
    output logic        clr_busy,

    // accumulate (from lbp_3x3_engine)
    input  logic        code_valid,
    input  logic [3:0]  blk_idx,
    input  logic [7:0]  lbp_code,
    output logic        overrun,

    // readout (from linear_svm_classifier)
    input  logic        rd_en,
    input  logic [11:0] rd_addr,
    output logic        rd_valid,
    output logic [14:0] rd_data
);
    localparam int PIX_LOG2 = 12;                  // 64x64 px per block
    localparam int Q_FRAC   = 14;
    localparam int SHIFT    = Q_FRAC - PIX_LOG2;   // 2
    localparam int CNT_W    = PIX_LOG2 + 1;        // 13 bits, max count 4096

    logic [CNT_W-1:0] mem [0:4095];
    logic [CNT_W-1:0] ram_q;                       // registered RAM read data

    logic [11:0] clr_cnt;
    logic [11:0] upd_addr;
    logic        upd_q;                            // write-back cycle of RMW

    wire [11:0] code_ad = {blk_idx, lbp_code};
    wire        take    = code_valid && !clr_busy && !rd_en && !upd_q;

    logic [11:0]       ram_addr;
    logic              ram_we;
    logic [CNT_W-1:0]  ram_wd;

    // priority: clear > readout > RMW write-back > RMW read
    always_comb begin
        ram_we   = 1'b0;
        ram_wd   = '0;
        ram_addr = code_ad;
        if (clr_busy) begin
            ram_addr = clr_cnt;
            ram_we   = 1'b1;
        end else if (rd_en) begin
            ram_addr = rd_addr;
        end else if (upd_q) begin
            ram_addr = upd_addr;
            ram_we   = 1'b1;
            ram_wd   = ram_q + 1'b1;
        end
    end

    always_ff @(posedge clk) begin
        if (ram_we) mem[ram_addr] <= ram_wd;
        ram_q <= mem[ram_addr];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            clr_busy <= 1'b0;
            clr_cnt  <= '0;
            upd_q    <= 1'b0;
            upd_addr <= '0;
            rd_valid <= 1'b0;
            overrun  <= 1'b0;
        end else begin
            if (clr_start) begin
                clr_busy <= 1'b1;
                clr_cnt  <= '0;
                overrun  <= 1'b0;
            end else if (clr_busy) begin
                clr_cnt <= clr_cnt + 12'd1;
                if (clr_cnt == 12'd4095) clr_busy <= 1'b0;
            end

            upd_q <= take;
            if (take) upd_addr <= code_ad;
            if (code_valid && !take) overrun <= 1'b1;

            rd_valid <= rd_en;
        end
    end

    assign rd_data = 15'(ram_q) << SHIFT;
endmodule
