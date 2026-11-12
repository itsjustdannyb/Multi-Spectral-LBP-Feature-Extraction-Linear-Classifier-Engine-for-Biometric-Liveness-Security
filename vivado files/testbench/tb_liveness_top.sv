`timescale 1ns/1ps
// Testbench for liveness_top. Generates a synthetic 256x256 test image
// (no image file needed), runs one full frame, prints the result.
// Needs svm_weights.mem added to the project as a simulation source.
module tb_liveness_top;
    logic        clk = 1'b0;
    logic        rst_n = 1'b0;
    logic        start = 1'b0;
    logic [7:0]  pix_data;
    logic        pix_valid;

    wire         pix_ready, busy, done, live, hist_overrun;
    wire         dbg_lbp_valid;
    wire signed [47:0] score;
    wire [7:0]   dbg_lbp_code;
    wire [3:0]   dbg_blk_idx;

    liveness_top dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .pix_data(pix_data), .pix_valid(pix_valid), .pix_ready(pix_ready),
        .busy(busy), .done(done), .live(live), .score(score),
        .hist_overrun(hist_overrun),
        .dbg_lbp_valid(dbg_lbp_valid), .dbg_lbp_code(dbg_lbp_code),
        .dbg_blk_idx(dbg_blk_idx)
    );

    always #5 clk = ~clk;                       // 100 MHz

    // ---- synthetic image, one pixel per index --------------------------------
    function automatic [7:0] pixel(input int i);
        int x, y;
        begin
            x = i % 256;
            y = i / 256;
            pixel = 8'((x*7 + y*13) ^ (x*y) ^ (x >> 2));
        end
    endfunction

    int pix_idx   = 0;
    int code_cnt  = 0;

    always_comb begin
        pix_valid = busy && (pix_idx < 65536);
        pix_data  = pixel(pix_idx < 65536 ? pix_idx : 0);
    end

    always @(posedge clk) begin
        if (pix_valid && pix_ready) pix_idx <= pix_idx + 1;
        if (dbg_lbp_valid)          code_cnt <= code_cnt + 1;
    end

    initial begin
        repeat (10) @(posedge clk);
        rst_n = 1'b1;
        repeat (5) @(posedge clk);
        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;

        @(posedge done);
        @(posedge clk);
        $display("---- frame finished ----");
        $display("pixels sent   : %0d (expect 65536)", pix_idx);
        $display("LBP codes     : %0d (expect 65536)", code_cnt);
        $display("hist_overrun  : %b (expect 0)", hist_overrun);
        $display("score         : %0d", score);
        $display("live          : %b", live);
        $finish;
    end

    initial begin                               // safety timeout, 3 ms
        #3_000_000;
        $display("TIMEOUT: done never asserted");
        $finish;
    end
endmodule
