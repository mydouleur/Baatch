module {
  aie.device(npu2_1col) {
    %logical_core = aie.logical_tile<CoreTile>(?, ?)
    %logical_shim_noc = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_0 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_1 = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @act(%logical_shim_noc, {%logical_core}, 2 : i32) : !aie.objectfifo<memref<512xf32>>  
    aie.objectfifo @out(%logical_core, {%logical_shim_noc_0}, 2 : i32) : !aie.objectfifo<memref<16xf32>>  
    aie.objectfifo @wts(%logical_shim_noc_1, {%logical_core}, 2 : i32) : !aie.objectfifo<memref<3136xui8>>  
    func.func private @xbf3_gemv_chunk(memref<3136xui8>, memref<512xf32>, memref<16xf32>) attributes {link_with = "xbf3_gemv_chunk.o"}
    %0 = aie.core(%logical_core) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %1 = aie.objectfifo.acquire @act(Consume, 1) : !aie.objectfifosubview<memref<512xf32>>
        %2 = aie.objectfifo.subview.access %1[0] : !aie.objectfifosubview<memref<512xf32>> -> memref<512xf32>
        %c0_2 = arith.constant 0 : index
        %c128 = arith.constant 128 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c128 step %c1_3 {
          %3 = aie.objectfifo.acquire @wts(Consume, 1) : !aie.objectfifosubview<memref<3136xui8>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<3136xui8>> -> memref<3136xui8>
          %5 = aie.objectfifo.acquire @out(Produce, 1) : !aie.objectfifosubview<memref<16xf32>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<16xf32>> -> memref<16xf32>
          func.call @xbf3_gemv_chunk(%4, %2, %6) : (memref<3136xui8>, memref<512xf32>, memref<16xf32>) -> ()
          aie.objectfifo.release @wts(Consume, 1)
          aie.objectfifo.release @out(Produce, 1)
        }
        aie.objectfifo.release @act(Consume, 1)
      }
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<401408xui8>, %arg1: memref<512xf32>, %arg2: memref<2048xf32>) {
      %1 = aiex.dma_configure_task_for @act {
        aie.dma_bd(%arg1 : memref<512xf32>, 0, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 512, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @wts {
        aie.dma_bd(%arg0 : memref<401408xui8>, 0, 401408, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 401408, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg2 : memref<2048xf32>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%3)
      aiex.dma_await_task(%3)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%2)
    }
  }
}

