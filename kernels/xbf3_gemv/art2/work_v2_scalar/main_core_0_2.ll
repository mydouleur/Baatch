; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2p"

@_anonymous0 = external global [3 x i32]
@act_cons_buff_1 = external global [512 x float]
@act_cons_buff_0 = external global [512 x float]
@out_buff_1 = external global [16 x float]
@out_buff_0 = external global [16 x float]
@wts_l2_cons_buff_1 = external global [50176 x i8]
@wts_l2_cons_buff_0 = external global [50176 x i8]
@wts_l1_cons_buff_1 = external global [3136 x i8]
@wts_l1_cons_buff_0 = external global [3136 x i8]

declare void @debug_i32(i32)

; Unknown intrinsic
declare void @llvm.aie2p.event(i32)

; Unknown intrinsic
declare void @llvm.aie2p.put.ms(i32, i32)

; Unknown intrinsic
declare { i32, i32 } @llvm.aie2p.get.ss()

; Unknown intrinsic
declare void @llvm.aie2p.mcd.write.vec(<16 x i32>, i32)

; Unknown intrinsic
declare <16 x i32> @llvm.aie2p.scd.read.vec(i32)

; Unknown intrinsic
declare void @llvm.aie2p.acquire(i32, i32)

; Unknown intrinsic
declare void @llvm.aie2p.release(i32, i32)

; Unknown intrinsic
declare void @llvm.aie2p.set.ctrl.reg(i32, i32)

declare void @xbf3_gemv_chunk(ptr, ptr, ptr)

define void @core_0_2() {
  store i32 0, ptr @_anonymous0, align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i64 4), align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i64 8), align 4
  br label %1

1:                                                ; preds = %38, %0
  %2 = phi i64 [ %44, %38 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775807
  br i1 %3, label %4, label %45

4:                                                ; preds = %1
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %5 = load i32, ptr @_anonymous0, align 4
  %6 = sext i32 %5 to i64
  switch i64 %6, label %7 [
    i64 0, label %46
    i64 1, label %48
  ]

7:                                                ; preds = %46, %48, %4
  %8 = phi ptr [ %49, %48 ], [ %47, %46 ], [ @act_cons_buff_0, %4 ]
  %9 = getelementptr [512 x float], ptr %8, i32 0, i32 0
  br label %10

10:                                               ; preds = %7
  br label %11

11:                                               ; preds = %26, %10
  %12 = phi i64 [ %37, %26 ], [ 0, %10 ]
  %13 = icmp slt i64 %12, 8192
  br i1 %13, label %14, label %38

14:                                               ; preds = %11
  call void @llvm.aie2p.acquire(i32 53, i32 -1)
  %15 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i64 4), align 4
  %16 = sext i32 %15 to i64
  switch i64 %16, label %17 [
    i64 0, label %50
    i64 1, label %52
  ]

17:                                               ; preds = %50, %52, %14
  %18 = phi ptr [ %53, %52 ], [ %51, %50 ], [ @wts_l1_cons_buff_0, %14 ]
  %19 = getelementptr [3136 x i8], ptr %18, i32 0, i32 0
  br label %20

20:                                               ; preds = %17
  call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %21 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i64 8), align 4
  %22 = sext i32 %21 to i64
  switch i64 %22, label %23 [
    i64 0, label %54
    i64 1, label %56
  ]

23:                                               ; preds = %54, %56, %20
  %24 = phi ptr [ %57, %56 ], [ %55, %54 ], [ @out_buff_0, %20 ]
  %25 = getelementptr [16 x float], ptr %24, i32 0, i32 0
  br label %26

26:                                               ; preds = %23
  call void @xbf3_gemv_chunk(ptr %19, ptr %9, ptr %25)
  call void @llvm.aie2p.release(i32 52, i32 1)
  %27 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i64 4), align 4
  %28 = add i32 %27, 1
  %29 = icmp sge i32 %28, 2
  %30 = add i32 %27, -1
  %31 = select i1 %29, i32 %30, i32 %28
  store i32 %31, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i64 4), align 4
  call void @llvm.aie2p.release(i32 51, i32 1)
  %32 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i64 8), align 4
  %33 = add i32 %32, 1
  %34 = icmp sge i32 %33, 2
  %35 = add i32 %32, -1
  %36 = select i1 %34, i32 %35, i32 %33
  store i32 %36, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i64 8), align 4
  %37 = add i64 %12, 1
  br label %11

38:                                               ; preds = %11
  call void @llvm.aie2p.release(i32 48, i32 1)
  %39 = load i32, ptr @_anonymous0, align 4
  %40 = add i32 %39, 1
  %41 = icmp sge i32 %40, 2
  %42 = add i32 %39, -1
  %43 = select i1 %41, i32 %42, i32 %40
  store i32 %43, ptr @_anonymous0, align 4
  %44 = add i64 %2, 1
  br label %1

45:                                               ; preds = %1
  ret void

46:                                               ; preds = %4
  %47 = phi ptr [ @act_cons_buff_0, %4 ]
  br label %7

48:                                               ; preds = %4
  %49 = phi ptr [ @act_cons_buff_1, %4 ]
  br label %7

50:                                               ; preds = %14
  %51 = phi ptr [ @wts_l1_cons_buff_0, %14 ]
  br label %17

52:                                               ; preds = %14
  %53 = phi ptr [ @wts_l1_cons_buff_1, %14 ]
  br label %17

54:                                               ; preds = %20
  %55 = phi ptr [ @out_buff_0, %20 ]
  br label %23

56:                                               ; preds = %20
  %57 = phi ptr [ @out_buff_1, %20 ]
  br label %23
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
