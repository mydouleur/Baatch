module {
  aie.device(npu2_1col) {
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %wts_cons_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "wts_cons_buff_0"} : memref<3136xui8> 
    %wts_cons_buff_1 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "wts_cons_buff_1"} : memref<3136xui8> 
    %wts_cons_prod_lock_0 = aie.lock(%tile_0_2, 4) {init = 2 : i32, sym_name = "wts_cons_prod_lock_0"}
    %wts_cons_cons_lock_0 = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "wts_cons_cons_lock_0"}
    %wts_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 4) {init = 0 : i32, sym_name = "wts_prod_lock_0"}
    %wts_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 5) {init = 0 : i32, sym_name = "wts_cons_lock_0"}
    %out_cons_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 0 : i32, sym_name = "out_cons_prod_lock_0"}
    %out_cons_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "out_cons_cons_lock_0"}
    %out_buff_0 = aie.buffer(%tile_0_2) {address = 4160 : i32, mem_bank = 0 : i32, sym_name = "out_buff_0"} : memref<16xf32> 
    %out_buff_1 = aie.buffer(%tile_0_2) {address = 19520 : i32, mem_bank = 1 : i32, sym_name = "out_buff_1"} : memref<16xf32> 
    %out_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "out_prod_lock_0"}
    %out_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "out_cons_lock_0"}
    %act_cons_buff_0 = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "act_cons_buff_0"} : memref<512xf32> 
    %act_cons_buff_1 = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "act_cons_buff_1"} : memref<512xf32> 
    %act_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "act_cons_prod_lock_0"}
    %act_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "act_cons_cons_lock_0"}
    %act_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 0 : i32, sym_name = "act_prod_lock_0"}
    %act_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "act_cons_lock_0"}
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %tile_0_2, DMA : 1)
    func.func private @xbf3_gemv_chunk(memref<3136xui8>, memref<512xf32>, memref<16xf32>) attributes {link_with = "xbf3_gemv_chunk.o"}
    %_anonymous0 = aie.buffer(%tile_0_2) {address = 34816 : i32, mem_bank = 2 : i32, sym_name = "_anonymous0"} : memref<3xi32> 
    %core_0_2 = aie.core(%tile_0_2) {
      %c1_i32 = arith.constant 1 : i32
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c128 = arith.constant 128 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2_i32 = arith.constant 2 : i32
      memref.store %c0_i32, %_anonymous0[%c0] : memref<3xi32>
      memref.store %c0_i32, %_anonymous0[%c1] : memref<3xi32>
      memref.store %c0_i32, %_anonymous0[%c2] : memref<3xi32>
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb17
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb18
    ^bb2:  // pred: ^bb1
      aie.use_lock(%act_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %2 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %3 = arith.index_cast %2 : i32 to index
      %4 = arith.index_cast %3 : index to i64
      cf.switch %4 : i64, [
        default: ^bb5,
        0: ^bb3,
        1: ^bb4
      ]
    ^bb3:  // pred: ^bb2
      cf.br ^bb6(%act_cons_buff_0 : memref<512xf32>)
    ^bb4:  // pred: ^bb2
      cf.br ^bb6(%act_cons_buff_1 : memref<512xf32>)
    ^bb5:  // pred: ^bb2
      cf.br ^bb6(%act_cons_buff_0 : memref<512xf32>)
    ^bb6(%5: memref<512xf32>):  // 3 preds: ^bb3, ^bb4, ^bb5
      cf.br ^bb7(%c0 : index)
    ^bb7(%6: index):  // 2 preds: ^bb6, ^bb16
      %7 = arith.cmpi slt, %6, %c128 : index
      cf.cond_br %7, ^bb8, ^bb17
    ^bb8:  // pred: ^bb7
      aie.use_lock(%wts_cons_cons_lock_0, AcquireGreaterEqual, 1)
      %8 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %9 = arith.index_cast %8 : i32 to index
      %10 = arith.index_cast %9 : index to i64
      cf.switch %10 : i64, [
        default: ^bb11,
        0: ^bb9,
        1: ^bb10
      ]
    ^bb9:  // pred: ^bb8
      cf.br ^bb12(%wts_cons_buff_0 : memref<3136xui8>)
    ^bb10:  // pred: ^bb8
      cf.br ^bb12(%wts_cons_buff_1 : memref<3136xui8>)
    ^bb11:  // pred: ^bb8
      cf.br ^bb12(%wts_cons_buff_0 : memref<3136xui8>)
    ^bb12(%11: memref<3136xui8>):  // 3 preds: ^bb9, ^bb10, ^bb11
      aie.use_lock(%out_prod_lock_0, AcquireGreaterEqual, 1)
      %12 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %13 = arith.index_cast %12 : i32 to index
      %14 = arith.index_cast %13 : index to i64
      cf.switch %14 : i64, [
        default: ^bb15,
        0: ^bb13,
        1: ^bb14
      ]
    ^bb13:  // pred: ^bb12
      cf.br ^bb16(%out_buff_0 : memref<16xf32>)
    ^bb14:  // pred: ^bb12
      cf.br ^bb16(%out_buff_1 : memref<16xf32>)
    ^bb15:  // pred: ^bb12
      cf.br ^bb16(%out_buff_0 : memref<16xf32>)
    ^bb16(%15: memref<16xf32>):  // 3 preds: ^bb13, ^bb14, ^bb15
      func.call @xbf3_gemv_chunk(%11, %5, %15) : (memref<3136xui8>, memref<512xf32>, memref<16xf32>) -> ()
      aie.use_lock(%wts_cons_prod_lock_0, Release, 1)
      %16 = memref.load %_anonymous0[%c1] : memref<3xi32>
      %17 = arith.addi %16, %c1_i32 : i32
      %18 = arith.cmpi sge, %17, %c2_i32 : i32
      %19 = arith.subi %17, %c2_i32 : i32
      %20 = arith.select %18, %19, %17 : i32
      memref.store %20, %_anonymous0[%c1] : memref<3xi32>
      aie.use_lock(%out_cons_lock_0, Release, 1)
      %21 = memref.load %_anonymous0[%c2] : memref<3xi32>
      %22 = arith.addi %21, %c1_i32 : i32
      %23 = arith.cmpi sge, %22, %c2_i32 : i32
      %24 = arith.subi %22, %c2_i32 : i32
      %25 = arith.select %23, %24, %22 : i32
      memref.store %25, %_anonymous0[%c2] : memref<3xi32>
      %26 = arith.addi %6, %c1 : index
      cf.br ^bb7(%26 : index)
    ^bb17:  // pred: ^bb7
      aie.use_lock(%act_cons_prod_lock_0, Release, 1)
      %27 = memref.load %_anonymous0[%c0] : memref<3xi32>
      %28 = arith.addi %27, %c1_i32 : i32
      %29 = arith.cmpi sge, %28, %c2_i32 : i32
      %30 = arith.subi %28, %c2_i32 : i32
      %31 = arith.select %29, %30, %28 : i32
      memref.store %31, %_anonymous0[%c0] : memref<3xi32>
      %32 = arith.addi %0, %c1 : index
      cf.br ^bb1(%32 : index)
    ^bb18:  // pred: ^bb1
      aie.end
    } {link_files = ["xbf3_gemv_chunk.o"]}
    aie.runtime_sequence(%arg0: memref<401408xui8>, %arg1: memref<512xf32>, %arg2: memref<2048xf32>) {
      %0 = aiex.dma_configure_task_for @act_shim_alloc {
        aie.dma_bd(%arg1 : memref<512xf32>, 0, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @wts_shim_alloc {
        aie.dma_bd(%arg0 : memref<401408xui8>, 0, 401408, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 401408, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @out_shim_alloc {
        aie.dma_bd(%arg2 : memref<2048xf32>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%2)
      aiex.dma_await_task(%2)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
    }
    aie.shim_dma_allocation @act_shim_alloc(%shim_noc_tile_0_0, MM2S, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%act_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%act_cons_buff_0 : memref<512xf32>, 0, 512) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%act_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%act_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%act_cons_buff_1 : memref<512xf32>, 0, 512) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%act_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%out_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buff_0 : memref<16xf32>, 0, 16) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%out_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%out_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buff_1 : memref<16xf32>, 0, 16) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%out_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%wts_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%wts_cons_buff_0 : memref<3136xui8>, 0, 3136) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%wts_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%wts_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%wts_cons_buff_1 : memref<3136xui8>, 0, 3136) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%wts_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @out_shim_alloc(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @wts_shim_alloc(%shim_noc_tile_0_0, MM2S, 1)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
  }
}
