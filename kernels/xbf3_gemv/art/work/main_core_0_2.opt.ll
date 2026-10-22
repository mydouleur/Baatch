; ModuleID = 'work/main_core_0_2.peano-compat.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@_anonymous0 = external local_unnamed_addr global [3 x i32]
@act_cons_buff_1 = external global [512 x float]
@act_cons_buff_0 = external global [512 x float]
@out_buff_1 = external global [16 x float]
@out_buff_0 = external global [16 x float]
@wts_cons_buff_1 = external global [3136 x i8]
@wts_cons_buff_0 = external global [3136 x i8]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.acquire(i32, i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.release(i32, i32) #0

declare void @xbf3_gemv_chunk(ptr, ptr, ptr) local_unnamed_addr

define void @core_0_2() local_unnamed_addr {
  store i32 0, ptr @_anonymous0, align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 4), align 4
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  br label %1

1:                                                ; preds = %0, %61
  %2 = phi i64 [ 0, %0 ], [ %67, %61 ]
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  %3 = load i32, ptr @_anonymous0, align 4
  %cond = icmp eq i32 %3, 1
  %spec.select = select i1 %cond, ptr @act_cons_buff_1, ptr @act_cons_buff_0
  br label %4

4:                                                ; preds = %4, %1
  %5 = phi i64 [ 0, %1 ], [ %59, %4 ]
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  %6 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 4), align 4
  %cond1 = icmp eq i32 %6, 1
  %spec.select4 = select i1 %cond1, ptr @wts_cons_buff_1, ptr @wts_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %7 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %cond2 = icmp eq i32 %7, 1
  %8 = select i1 %cond2, ptr @out_buff_1, ptr @out_buff_0
  tail call void @xbf3_gemv_chunk(ptr nonnull %spec.select4, ptr nonnull %spec.select, ptr nonnull %8)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  %9 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 4), align 4
  %10 = add i32 %9, 1
  %11 = icmp sgt i32 %10, 1
  %12 = add i32 %9, -1
  %13 = select i1 %11, i32 %12, i32 %10
  store i32 %13, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 4), align 4
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %14 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %15 = add i32 %14, 1
  %16 = icmp sgt i32 %15, 1
  %17 = add i32 %14, -1
  %18 = select i1 %16, i32 %17, i32 %15
  store i32 %18, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  %19 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 4), align 4
  %cond1.1 = icmp eq i32 %19, 1
  %spec.select4.1 = select i1 %cond1.1, ptr @wts_cons_buff_1, ptr @wts_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %20 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %cond2.1 = icmp eq i32 %20, 1
  %21 = select i1 %cond2.1, ptr @out_buff_1, ptr @out_buff_0
  tail call void @xbf3_gemv_chunk(ptr nonnull %spec.select4.1, ptr nonnull %spec.select, ptr nonnull %21)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  %22 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 4), align 4
  %23 = add i32 %22, 1
  %24 = icmp sgt i32 %23, 1
  %25 = add i32 %22, -1
  %26 = select i1 %24, i32 %25, i32 %23
  store i32 %26, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 4), align 4
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %27 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %28 = add i32 %27, 1
  %29 = icmp sgt i32 %28, 1
  %30 = add i32 %27, -1
  %31 = select i1 %29, i32 %30, i32 %28
  store i32 %31, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  %32 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 4), align 4
  %cond1.2 = icmp eq i32 %32, 1
  %spec.select4.2 = select i1 %cond1.2, ptr @wts_cons_buff_1, ptr @wts_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %33 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %cond2.2 = icmp eq i32 %33, 1
  %34 = select i1 %cond2.2, ptr @out_buff_1, ptr @out_buff_0
  tail call void @xbf3_gemv_chunk(ptr nonnull %spec.select4.2, ptr nonnull %spec.select, ptr nonnull %34)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  %35 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 4), align 4
  %36 = add i32 %35, 1
  %37 = icmp sgt i32 %36, 1
  %38 = add i32 %35, -1
  %39 = select i1 %37, i32 %38, i32 %36
  store i32 %39, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 4), align 4
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %40 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %41 = add i32 %40, 1
  %42 = icmp sgt i32 %41, 1
  %43 = add i32 %40, -1
  %44 = select i1 %42, i32 %43, i32 %41
  store i32 %44, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %45 = or disjoint i64 %5, 3
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  %46 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 4), align 4
  %cond1.3 = icmp eq i32 %46, 1
  %spec.select4.3 = select i1 %cond1.3, ptr @wts_cons_buff_1, ptr @wts_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %47 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %cond2.3 = icmp eq i32 %47, 1
  %48 = select i1 %cond2.3, ptr @out_buff_1, ptr @out_buff_0
  tail call void @xbf3_gemv_chunk(ptr nonnull %spec.select4.3, ptr nonnull %spec.select, ptr nonnull %48)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  %49 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 4), align 4
  %50 = add i32 %49, 1
  %51 = icmp sgt i32 %50, 1
  %52 = add i32 %49, -1
  %53 = select i1 %51, i32 %52, i32 %50
  store i32 %53, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 4), align 4
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %54 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %55 = add i32 %54, 1
  %56 = icmp sgt i32 %55, 1
  %57 = add i32 %54, -1
  %58 = select i1 %56, i32 %57, i32 %55
  store i32 %58, ptr getelementptr inbounds nuw (i8, ptr @_anonymous0, i20 8), align 4
  %59 = add nuw nsw i64 %5, 4
  %60 = icmp samesign ult i64 %45, 127
  br i1 %60, label %4, label %61

61:                                               ; preds = %4
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  %62 = load i32, ptr @_anonymous0, align 4
  %63 = add i32 %62, 1
  %64 = icmp sgt i32 %63, 1
  %65 = add i32 %62, -1
  %66 = select i1 %64, i32 %65, i32 %63
  store i32 %66, ptr @_anonymous0, align 4
  %67 = add nuw nsw i64 %2, 1
  %.not = icmp eq i64 %67, 9223372036854775807
  br i1 %.not, label %68, label %1

68:                                               ; preds = %61
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
