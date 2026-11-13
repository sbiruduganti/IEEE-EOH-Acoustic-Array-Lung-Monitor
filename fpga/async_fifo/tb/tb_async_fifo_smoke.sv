// =============================================================================
// tb_async_fifo_smoke.sv -- SMOKE test only (not the real verification plan)
//
// Unrelated clocks, random push/pop, scoreboard checks data order & integrity,
// plus a fill-to-full and drain-to-empty phase. Proves v1 is not broken.
// Real verification (SVA, coverage, reset-mid-traffic, clock-ratio sweeps)
// is still TODO.
//
// Run (Icarus):  iverilog -g2012 -o sim rtl/sync_2ff.sv rtl/async_fifo.sv tb/tb_async_fifo_smoke.sv && vvp sim
// =============================================================================
`timescale 1ns/1ps

module tb_async_fifo_smoke;

  localparam int DW = 16;
  localparam int AW = 4;          // depth 16
  localparam int N  = 5000;       // words to push in random phase

  logic wr_clk = 0, rd_clk = 0;
  logic wr_rst_n = 0, rd_rst_n = 0;
  logic wr_en = 0, rd_en = 0;
  logic [DW-1:0] wr_data = '0, rd_data;
  logic wr_full, rd_empty;

  // unrelated clocks: 100 MHz write, ~73 MHz read
  always #5.0   wr_clk = ~wr_clk;
  always #6.85  rd_clk = ~rd_clk;

  async_fifo #(.DATA_WIDTH(DW), .ADDR_WIDTH(AW)) dut (.*);

  // simple scoreboard (expected data is a counter sequence)
  int wr_count = 0, rd_count = 0, errors = 0;

  task automatic push_random(input int n, input int pct);
    while (wr_count < n) begin
      @(posedge wr_clk);
      if (wr_en && !wr_full) begin
        wr_count++;
        wr_data <= wr_data + 1'b1;
      end
      wr_en <= ($urandom_range(99) < pct) && (wr_count < n);
    end
    @(posedge wr_clk); wr_en <= 0;
  endtask

  // reader: checks every popped word against expected counter value
  always @(posedge rd_clk) begin
    if (rd_en && !rd_empty) begin
      if (rd_data !== DW'(rd_count)) begin
        errors++;
        if (errors < 10)
          $display("ERROR @%0t: expected %0d got %0d", $time, rd_count, rd_data);
      end
      rd_count++;
    end
  end

  // overflow / underflow sanity: count can never exceed DEPTH
  always @(posedge wr_clk)
    if (wr_rst_n && (wr_count - rd_count) > (1 << AW)) begin
      errors++; $display("ERROR: occupancy exceeded depth");
    end

  initial begin
    // reset both domains together
    repeat (5) @(posedge wr_clk);
    wr_rst_n = 1; rd_rst_n = 1;
    repeat (3) @(posedge wr_clk);

    // ---- phase 1: random traffic both sides ----
    fork
      push_random(N, 60);
      begin
        while (rd_count < N) begin
          @(posedge rd_clk);
          rd_en <= ($urandom_range(99) < 55);
        end
        rd_en <= 0;
      end
    join
    $display("phase 1 random:  pushed %0d, popped %0d", wr_count, rd_count);

    // ---- phase 2: fill until full (reader stopped) ----
    rd_en <= 0;
    begin
      int start;
      start = wr_count;
      // hold wr_en high for 40 cycles; FIFO must accept exactly 16 then stall
      repeat (40) begin
        @(posedge wr_clk);
        if (wr_en && !wr_full) begin
          wr_count++;
          wr_data <= wr_data + 1'b1;
        end
        wr_en <= 1;
      end
      @(posedge wr_clk);
      if (wr_en && !wr_full) begin wr_count++; wr_data <= wr_data + 1'b1; end
      wr_en <= 0;
      repeat (10) @(posedge wr_clk);
      if (!wr_full) begin errors++; $display("ERROR: not full after filling"); end
      if (wr_count - start != (1 << AW)) begin
        errors++; $display("ERROR: accepted %0d words, expected %0d", wr_count-start, 1<<AW);
      end
      $display("phase 2 fill:    accepted %0d words then full=%0b", wr_count-start, wr_full);
    end

    // ---- phase 3: drain until empty ----
    while (rd_count < wr_count) begin
      @(posedge rd_clk); rd_en <= 1;
    end
    @(posedge rd_clk); rd_en <= 0;
    repeat (10) @(posedge rd_clk);
    if (!rd_empty) begin errors++; $display("ERROR: not empty after draining"); end
    $display("phase 3 drain:   popped %0d total, empty=%0b", rd_count, rd_empty);

    if (errors == 0) $display("SMOKE TEST PASSED");
    else             $display("SMOKE TEST FAILED: %0d errors", errors);
    $finish;
  end

  initial begin #5ms; $display("TIMEOUT"); $finish; end

endmodule
