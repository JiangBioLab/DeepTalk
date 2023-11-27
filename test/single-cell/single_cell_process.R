library(foreach)
library(Matrix)

setClass("DeepTalk", representation(data = "list", meta = "list", para = "list", coef = "matrix",
    cellpair = "list", dist = "matrix", lrpair = "data.frame", tf = "data.frame",
    lr_path = "list"), prototype(data = list(), meta = list(), para = list(), coef = matrix(),
    cellpair = list(), dist = matrix(), lrpair = data.frame(), tf = data.frame(),
    lr_path = list()))


.rename_chr <- function(x){
    x <- strsplit(x, split = " ")
    x_new <- NULL
    for (i in 1:length(x)) {
        x1 <- x[[i]]
        if (length(x1) > 1) {
            x2 <- x1[1]
            for (j in 2:length(x1)) {
                x2 <- paste(x2, x1[j], sep = "_")
            }
            x1 <- x2
        }
        x_new <- c(x_new, x1)
    }
    x <- strsplit(x_new, split = "-")
    x_new <- NULL
    for (i in 1:length(x)) {
        x1 <- x[[i]]
        if (length(x1) > 1) {
            x2 <- x1[1]
            for (j in 2:length(x1)) {
                x2 <- paste(x2, x1[j], sep = "_")
            }
            x1 <- x2
        }
        x_new <- c(x_new, x1)
    }
    return(x_new)
}
.get_cellpair1 <- function(celltype_dist, st_meta, celltype_sender, celltype_receiver, n_neighbor) {
    cell_sender <- st_meta[st_meta$celltype == celltype_sender, ]
    cell_receiver <- st_meta[st_meta$celltype == celltype_receiver, ]
    cell_pair <- list()
    for (j in 1:nrow(cell_sender)) {
        celltype_dist1 <- celltype_dist[, cell_sender$cell[j]]
        celltype_dist1 <- celltype_dist1[celltype_dist1 > 0]
        celltype_dist1 <- celltype_dist1[order(celltype_dist1)]
        #celltype_dist1 <- (celltype_dist1[1:100])
        cell_pair[[j]] <- names(celltype_dist1[celltype_dist1<200])
    }
    names(cell_pair) <- cell_sender$cell
    cell_pair <- reshape2::melt(cell_pair)
    colnames(cell_pair)<-c("cell_receiver","cell_sender")
    cell_pair<-cell_pair[, c("cell_sender","cell_receiver")]
    cell_pair$cell_sender <- as.character(cell_pair$cell_sender)
    cell_pair <- cell_pair[cell_pair$cell_receiver %in% cell_receiver$cell, ]
    return(cell_pair)
}
.lr_distance1 <- function(st_data, cell_pair, lrdb, celltype_sender, celltype_receiver, per_num, pvalue) {
    .co_exp <- function(x) {
        x_1 <- x[1:(length(x)/2)]
        x_2 <- x[(length(x)/2 + 1):length(x)]
        x_12 <- x_1 * x_2
        x_12_ratio <- length(x_12[x_12 > 0])/length(x_12)
        return(x_12_ratio)
    }
    .co_exp_p <- function(x) {
        x_real <- x[length(x)]
        x_per <- x[-length(x)]
        x_p <- length(x_per[x_per >= x_real])/length(x_per)
    }
    ### [1] LR distance
    lrdb$celltype_sender <- celltype_sender
    lrdb$celltype_receiver <- celltype_receiver
    lrdb$lr_co_exp_num <- 0
    lrdb$lr_co_ratio <- 0
    lrdb$lr_co_ratio_pvalue <- 1
    rownames(lrdb) <- 1:nrow(lrdb)
    # calculate co-expression ratio
    if (nrow(lrdb) == 1) {
        ndata_lr <- ndata_ligand * ndata_receptor
        lrdb$lr_co_exp_num <- length(ndata_lr[ndata_lr > 0])
        lrdb$lr_co_ratio <- length(ndata_lr[ndata_lr > 0])/length(ndata_lr)
    } else {
        ndata_ligand <- st_data[lrdb$ligand, cell_pair$cell_sender]
        ndata_receptor <- st_data[lrdb$receptor, cell_pair$cell_receiver]
        ndata_lr <- cbind(ndata_ligand, ndata_receptor)
        lrdb$lr_co_ratio <- apply(ndata_lr, 1, .co_exp)
        lrdb$lr_co_exp_num <- apply(ndata_lr, 1, .co_exp) * nrow(cell_pair)
    }
    # permutation test
    res_per <- list()
    for (j in 1:per_num) {
        print(j)
        set.seed(j)
        cell_id <- sample(x = 1:ncol(st_data), size = nrow(cell_pair) * 2, replace = T)
        ndata_ligand <- st_data[lrdb$ligand, cell_id[1:(length(cell_id)/2)]]
        ndata_receptor <- st_data[lrdb$receptor, cell_id[(length(cell_id)/2 + 1):length(cell_id)]]
        if (nrow(lrdb) == 1) {
            ndata_lr <- ndata_ligand * ndata_receptor
            res_per[[j]] <- length(ndata_lr[ndata_lr > 0])/length(ndata_lr)
        } else {
            ndata_lr <- cbind(ndata_ligand, ndata_receptor)
            res_per[[j]] <- apply(ndata_lr, 1, .co_exp)
        }
    }
    names(res_per) <- paste0("P", 1:length(res_per))
    res_per <- as.data.frame(res_per)
    res_per$real <- lrdb$lr_co_ratio
    lrdb$lr_co_ratio_pvalue <- apply(res_per, 1, .co_exp_p)
    lrdb <- lrdb[lrdb$lr_co_ratio_pvalue < pvalue, ]
    return(lrdb)
}

dec_cci_all1 <- function(object, n_neighbor = 10, min_pairs = 5, min_pairs_ratio = 0,
    per_num = 1000, pvalue = 0.05, co_exp_ratio = 0.1, if_doParallel = T, use_n_cores = NULL) {
    # check input data

    if (is.null(use_n_cores)) {
        n_cores <- parallel::detectCores()
        n_cores <- floor(n_cores/4)
        if (n_cores < 2) {
            if_doParallel <- FALSE
        }
    } else {
        n_cores <- use_n_cores
    }
    if (if_doParallel) {
        cl <- parallel::makeCluster(n_cores)
        doParallel::registerDoParallel(cl)
    }
    st_meta <- .get_st_meta(object)
    st_data <- .get_st_data(object)
    celltype_dist <- object@dist
    # generate pair-wise cell types
    cellname <- unique(st_meta$celltype)
    celltype_pair <- NULL
    for (i in 1:length(cellname)) {
        d1 <- data.frame(celltype_sender = rep(cellname[i], length(cellname)), celltype_receiver = cellname,
            stringsAsFactors = F)
        celltype_pair <- rbind(celltype_pair, d1)
    }
    celltype_pair <- celltype_pair[celltype_pair$celltype_sender != celltype_pair$celltype_receiver, ]
    #cat(paste0("Note: there are ", length(cellname), " cell types and ", nrow(celltype_pair), " pair-wise cell pairs"), "\n")
    pathways <- object@lr_path$pathways
    ggi_tf <- unique(pathways[, c("src", "dest", "src_tf", "dest_tf")])
    #cat(crayon::cyan("Begin to find LR pairs", "\n"))
    if (if_doParallel) {
        all_res <- foreach::foreach(i=1:nrow(celltype_pair), .packages = c("Matrix", "reshape2"),
            .export = c(".get_cellpair1", ".lr_distance1", ".get_tf_res", ".get_score")) %dopar% {
            celltype_sender <- celltype_pair$celltype_sender[i]
            celltype_receiver <- celltype_pair$celltype_receiver[i]
            cell_pair <- .get_cellpair1(celltype_dist, st_meta, celltype_sender, celltype_receiver, n_neighbor)
            cell_sender <- st_meta[st_meta$celltype == celltype_sender, ]
            cell_receiver <- st_meta[st_meta$celltype == celltype_receiver, ]
            cell_pair_all <- nrow(cell_sender) * nrow(cell_receiver)/2
            if (nrow(cell_pair) > min_pairs) {
                if (nrow(cell_pair) > cell_pair_all * min_pairs_ratio) {
                    lrdb <- object@lr_path$lrpairs
                    ### [1] LR distance
                    lrdb <- .lr_distance1(st_data, cell_pair, lrdb, celltype_sender, celltype_receiver, per_num, pvalue)
                    ### [2] Downstream targets and TFs
                    max_hop <- object@para$max_hop
                    receptor_tf <- NULL
                    if (nrow(lrdb) > 0) {
                        receptor_tf <- .get_tf_res(celltype_sender, celltype_receiver, lrdb, ggi_tf, cell_pair, st_data, max_hop, co_exp_ratio)
                        if (!is.null(receptor_tf)) {
                            # calculate score
                            lrdb <- .get_score(lrdb, receptor_tf)
                        } else {
                            lrdb <- NULL
                        }
                    }
                    if (is.data.frame(lrdb)) {
                        if (nrow(lrdb) > 0) {
                            list(lrdb = lrdb, receptor_tf=receptor_tf, cell_pair=cell_pair)
                        }
                    }
                }
            }
        }
        doParallel::stopImplicitCluster()
        parallel::stopCluster(cl)
        res_receptor_tf <- NULL
        res_lrpair <- NULL
        res_cellpair <- list()
        m <- 0
        for (i in 1:length(all_res)) {
            all_res1 <- all_res[[i]]
            if (!is.null(all_res1)) {
                m <- m+1
                res_lrpair <- rbind(res_lrpair, all_res1[[1]])
                res_receptor_tf <- rbind(res_receptor_tf, all_res1[[2]])
                res_cellpair[[m]] <- all_res1[[3]]
                names(res_cellpair)[m] <- paste0(unique(all_res1[[1]]$celltype_sender), " -- ", unique(all_res1[[1]]$celltype_receiver))
            }
        }
        if (!is.null(res_lrpair)) {
            object@lrpair <- res_lrpair
        }
        if (!is.null(res_receptor_tf)) {
            object@tf <- res_receptor_tf
        }
        if (length(res_cellpair) > 0) {
            object@cellpair <- res_cellpair
        }
    } else {
        for (i in 1:nrow(celltype_pair)) {
            celltype_sender <- celltype_pair$celltype_sender[i]
            celltype_receiver <- celltype_pair$celltype_receiver[i]
            cell_pair <- .get_cellpair1(celltype_dist, st_meta, celltype_sender, celltype_receiver, n_neighbor)
            cell_sender <- st_meta[st_meta$celltype == celltype_sender, ]
            cell_receiver <- st_meta[st_meta$celltype == celltype_receiver, ]
            cell_pair_all <- nrow(cell_sender) * nrow(cell_receiver)/2
            if (nrow(cell_pair) > min_pairs) {
                if (nrow(cell_pair) > cell_pair_all * min_pairs_ratio) {
                    lrdb <- object@lr_path$lrpairs
                    ### [1] LR distance
                    lrdb <- .lr_distance1(st_data, cell_pair, lrdb, celltype_sender, celltype_receiver, per_num, pvalue)
                    ### [2] Downstream targets and TFs
                    max_hop <- object@para$max_hop
                    receptor_tf <- NULL
                    if (nrow(lrdb) > 0) {
                        receptor_tf <- .get_tf_res(celltype_sender, celltype_receiver, lrdb, ggi_tf, cell_pair, st_data, max_hop, co_exp_ratio)
                        if (!is.null(receptor_tf)) {
                            # calculate score
                            lrdb <- .get_score(lrdb, receptor_tf)
                        } else {
                            lrdb <- "NA"
                        }
                    }
                    lrpair <- object@lrpair
                    if (nrow(lrpair) == 0) {
                        if (is.data.frame(lrdb)) {
                            object@lrpair <- lrdb
                        }
                    } else {
                        if (is.data.frame(lrdb)) {
                            lrpair <- rbind(lrpair, lrdb)
                            object@lrpair <- unique(lrpair)
                        }
                    }
                    tf <- object@tf
                    if (nrow(tf) == 0) {
                        if (is.data.frame(receptor_tf)) {
                            object@tf <- receptor_tf
                        }
                    } else {
                        if (is.data.frame(receptor_tf)) {
                            tf <- rbind(tf, receptor_tf)
                            object@tf <- unique(tf)
                        }
                    }
                    object@cellpair[[paste0(celltype_sender, " -- ", celltype_receiver)]] <- cell_pair
                }
            }
            sym <- crayon::combine_styles("bold", "green")
            cat(sym("***Done***"), paste0(celltype_sender, " -- ", celltype_receiver),"\n")
        }
    }
    object@para$min_pairs <- min_pairs
    object@para$min_pairs_ratio <- min_pairs_ratio
    object@para$per_num <- per_num
    object@para$pvalue <- pvalue
    object@para$co_exp_ratio <- co_exp_ratio
    return(object)
}

.percent_cell <- function(x) {
    return(length(x[x > 0]))
}
.show_warning <- function(celltype, celltype_new){
    sc_meta <- data.frame(celltype = celltype, celltype_new = celltype_new, stringsAsFactors = FALSE)
    sc_meta <- unique(sc_meta)
    sc_meta <- sc_meta[sc_meta$celltype != sc_meta$celltype_new, ]
    warning_info <- NULL
    if (nrow(sc_meta) > 0) {
        warning_info <- "celltype of "
        if (nrow(sc_meta) == 1) {
            warning_info <- paste0(warning_info, sc_meta$celltype[1], " has been replaced by ", sc_meta$celltype_new[1])
        } else {
            for (i in 1:nrow(sc_meta)) {
                if (i == nrow(sc_meta)) {
                    warning_info <- paste0(warning_info, "and ",sc_meta$celltype[i])
                } else {
                    warning_info <- paste0(warning_info, sc_meta$celltype[i], ", ")
                }
            }
            warning_info <- paste0(warning_info, " have been replaced by ")
            for (i in 1:nrow(sc_meta)) {
                if (i == nrow(sc_meta)) {
                    warning_info <- paste0(warning_info, "and ",sc_meta$celltype_new[i])
                } else {
                    warning_info <- paste0(warning_info, sc_meta$celltype_new[i], ", ")
                }
            }
        }
    }
    return(warning_info)
}
.get_st_meta <- function(object) {
    st_type <- object@para$st_type
    if_skip_dec_celltype <- object@para$if_skip_dec_celltype
    if (st_type == "single-cell") {
        st_meta <- object@meta$rawmeta
        st_meta <- st_meta[st_meta$celltype != "unsure", ]
        st_meta <- st_meta[st_meta$label != "less nFeatures", ]
        if (nrow(st_meta) == 0) {
            stop("No cell types found in rawmeta by excluding the unsure and less nFeatures cells!")
        }
    } else {
        if (if_skip_dec_celltype) {
            st_meta <- object@meta$rawmeta
            colnames(st_meta)[1] <- "cell"
        } else {
            st_meta <- object@meta$newmeta
        }
    }
    return(st_meta)
}
.get_st_data <- function(object) {
    st_type <- object@para$st_type
    if_skip_dec_celltype <- object@para$if_skip_dec_celltype
    if (st_type == "single-cell") {
        st_data <- object@data
        if (if_skip_dec_celltype) {
            st_data <- st_data$rawdata
        } else {
            st_data <- st_data$rawndata
        }
        st_meta <- object@meta$rawmeta
        st_meta <- st_meta[st_meta$celltype != "unsure", ]
        st_meta <- st_meta[st_meta$label != "less nFeatures", ]
        st_data <- st_data[, st_meta$cell]
        gene_expressed_ratio <- rowSums(st_data)
        st_data <- st_data[which(gene_expressed_ratio > 0), ]
        if (nrow(st_data) == 0) {
            stop("No cell types found in rawmeta by excluding the unsure and less nFeatures cells!")
        }
    } else {
        if (if_skip_dec_celltype) {
            st_data <- object@data$rawdata
        } else {
            st_data <- object@data$newdata
        }
        gene_expressed_ratio <- rowSums(st_data)
        st_data <- st_data[which(gene_expressed_ratio > 0), ]
        if (nrow(st_data) == 0) {
            stop("No expressed genes in newdata!")
        }
    }
    return(st_data)
}
.st_dist <- function(st_meta) {
    st_dist <- as.matrix(stats::dist(x = cbind(st_meta$x, st_meta$y)))
    rownames(st_dist) <- st_meta[, 1]
    colnames(st_dist) <- st_meta[, 1]
    return(st_dist)
}
.get_tf_res <- function(celltype_sender, celltype_receiver, lrdb, ggi_tf, cell_pair, st_data, max_hop, co_exp_ratio) {
    .co_exp <- function(x) {
        x_1 <- x[1:(length(x)/2)]
        x_2 <- x[(length(x)/2 + 1):length(x)]
        x_12 <- x_1 * x_2
        x_12_ratio <- length(x_12[x_12 > 0])/length(x_12)
        return(x_12_ratio)
    }
    .co_exp_batch <- function(st_data, ggi_res, cell_pair) {
        ggi_res_temp <- unique(ggi_res[, c("src", "dest")])
        cell_receiver <- unique(cell_pair$cell_receiver)
        m <- floor(nrow(ggi_res_temp)/5000)
        i <- 1
        res <- NULL
        while (i <= (m + 1)) {
            m_int <- 5000 * i
            if (m_int < nrow(ggi_res_temp)) {
                ggi_res_temp1 <- ggi_res_temp[((i - 1) * 5000 + 1):(5000 * i), ]
            } else {
                if (m_int == nrow(ggi_res_temp)) {
                    ggi_res_temp1 <- ggi_res_temp[((i - 1) * 5000 + 1):(5000 * i), ]
                    i <- i + 1
                } else {
                    ggi_res_temp1 <- ggi_res_temp[((i - 1) * 5000 + 1):nrow(ggi_res_temp), ]
                }
            }
            ndata_src <- st_data[ggi_res_temp1$src, cell_receiver]
            ndata_dest <- st_data[ggi_res_temp1$dest, cell_receiver]
            ndata_gg <- cbind(ndata_src, ndata_dest)
            # calculate co-expression
            ggi_res_temp1$co_ratio <- NA
            ggi_res_temp1$co_ratio <- apply(ndata_gg, 1, .co_exp)
            res <- rbind(res, ggi_res_temp1)
            i <- i + 1
        }
        res$merge_key <- paste0(res$src, res$dest)
        ggi_res$merge_key <- paste0(ggi_res$src, ggi_res$dest)
        ggi_res <- merge(ggi_res, res, by = "merge_key", all.x = T, sort = F)
        ggi_res <- ggi_res[, c(2:6, 9)]
        colnames(ggi_res) <- c("src", "dest", "src_tf", "dest_tf", "hop", "co_ratio")
        return(ggi_res)
    }
    .generate_ggi_res <- function(ggi_tf, cell_pair, receptor_name, st_data, max_hop, co_exp_ratio) {
        # generate ggi_res
        ggi_res <- NULL
        ggi_tf1 <- ggi_tf[ggi_tf$src == receptor_name, ]
        ggi_tf1 <- unique(ggi_tf1[ggi_tf1$dest %in% rownames(st_data), ])
        n <- 0
        ggi_tf1$hop <- n + 1
        while (n <= max_hop) {
            ggi_res <- rbind(ggi_res, ggi_tf1)
            ggi_tf1 <- ggi_tf[ggi_tf$src %in% ggi_tf1$dest, ]
            ggi_tf1 <- unique(ggi_tf1[ggi_tf1$dest %in% rownames(st_data), ])
            if (nrow(ggi_tf1) == 0) {
                break
            }
            ggi_tf1$hop <- n + 2
            n <- n + 1
        }
        ggi_res <- unique(ggi_res)
        # ndata_src and ndata_dest
        ggi_res_temp <- unique(ggi_res[, c("src", "dest")])
        if (nrow(ggi_res_temp) >= 5000) {
            ggi_res <- .co_exp_batch(st_data, ggi_res, cell_pair)
        } else {
            ndata_src <- st_data[ggi_res$src, cell_pair$cell_receiver]
            ndata_dest <- st_data[ggi_res$dest, cell_pair$cell_receiver]
            ndata_gg <- cbind(ndata_src, ndata_dest)
            # calculate co-expression
            ggi_res$co_ratio <- NA
            ggi_res$co_ratio <- apply(ndata_gg, 1, .co_exp)
        }
        ggi_res <- ggi_res[ggi_res$co_ratio > co_exp_ratio, ]
        return(ggi_res)
    }
    .generate_tf_gene_all <- function(ggi_res, max_hop) {
        tf_gene_all <- NULL
        ggi_hop <- ggi_res[ggi_res$hop == 1, ]
        for (k in 1:max_hop) {
            ggi_hop_yes <- ggi_hop[ggi_hop$dest_tf == "YES", ]
            if (nrow(ggi_hop_yes) > 0) {
                ggi_hop_tf <- ggi_res[ggi_res$hop == k + 1, ]
                if (nrow(ggi_hop_tf) > 0) {
                    ggi_hop_yes <- ggi_hop_yes[ggi_hop_yes$dest %in% ggi_hop_tf$src, ]
                    if (nrow(ggi_hop_yes) > 0) {
                        tf_gene <- ggi_hop_yes$hop
                        names(tf_gene) <- ggi_hop_yes$dest
                        tf_gene_all <- c(tf_gene_all, tf_gene)
                    }
                }
            }
            ggi_hop_no <- ggi_hop[ggi_hop$dest_tf == "NO", ]
            ggi_hop <- ggi_res[ggi_res$hop == k + 1, ]
            ggi_hop <- ggi_hop[ggi_hop$src %in% ggi_hop_no$dest, ]
        }
        return(tf_gene_all)
    }
    .generate_tf_res <- function(tf_gene_all, celltype_sender, celltype_receiver, receptor_name, ggi_res) {
        receptor_tf_temp <- data.frame(celltype_sender = celltype_sender, celltype_receiver = celltype_receiver,
                                       receptor = receptor_name, tf = names(tf_gene_all), n_hop = as.numeric(tf_gene_all), n_target = 0, stringsAsFactors = F)
        tf_names <- names(tf_gene_all)
        tf_n_hop <- as.numeric(tf_gene_all)
        for (i in 1:length(tf_names)) {
            ggi_res_tf <- ggi_res[ggi_res$src == tf_names[i] & ggi_res$hop == tf_n_hop[i] + 1, ]
            receptor_tf_temp$n_target[i] <- length(unique(ggi_res_tf$dest))
        }
        return(receptor_tf_temp)
    }
    .random_walk <- function(receptor_tf, ggi_res) {
        receptor_name <- unique(receptor_tf$receptor)
        tf_score <- rep(0, nrow(receptor_tf))
        names(tf_score) <- receptor_tf$tf
        max_n_hop <- 10
        for (i in 1:10000) {
            ggi_res1 <- ggi_res[ggi_res$src == receptor_name, ]
            n_hop <- 1
            while (nrow(ggi_res1) > 0 & n_hop <= max_n_hop) {
                target_name <- sample(x = 1:nrow(ggi_res1), size = 1)
                ggi_res1 <- ggi_res1[target_name, ]
                if (ggi_res1$dest %in% names(tf_score)) {
                    tf_score[ggi_res1$dest] <- tf_score[ggi_res1$dest] + 1
                }
                ggi_res1 <- ggi_res[ggi_res$src == ggi_res1$dest, ]
                n_hop <- n_hop + 1
            }
        }
        tf_score <- as.numeric(tf_score/10000)
    }
    receptor_tf <- NULL
    receptor_name <- unique(lrdb$receptor)
    for (j in 1:length(receptor_name)) {
        # generate ggi_res
        ggi_res <- .generate_ggi_res(ggi_tf, cell_pair, receptor_name[j], st_data, max_hop, co_exp_ratio)
        if (nrow(ggi_res) > 0) {
            tf_gene_all <- .generate_tf_gene_all(ggi_res, max_hop)
            tf_gene_all <- data.frame(gene = names(tf_gene_all), hop = tf_gene_all, stringsAsFactors = F)
            tf_gene_all_new <- unique(tf_gene_all)
            tf_gene_all <- tf_gene_all_new$hop
            names(tf_gene_all) <- tf_gene_all_new$gene
            ggi_res$dest_tf_enrich <- "NO"
            if (!is.null(tf_gene_all)) {
                ggi_res[ggi_res$dest %in% names(tf_gene_all), ]$dest_tf_enrich <- "YES"
                # generate tf res
                receptor_tf_temp <- .generate_tf_res(tf_gene_all, celltype_sender, celltype_receiver, receptor_name[j], ggi_res)
                # random walk
                receptor_tf_temp$score <- .random_walk(receptor_tf_temp, ggi_res)
                receptor_tf <- rbind(receptor_tf, receptor_tf_temp)
            }
        }
    }
    return(receptor_tf)
}

.random_walk <- function(receptor_tf, ggi_res) {
    receptor_name <- unique(receptor_tf$receptor)
    tf_score <- rep(0, nrow(receptor_tf))
    names(tf_score) <- receptor_tf$tf
    max_n_hop <- 10
    for (i in 1:10000) {
        ggi_res1 <- ggi_res[ggi_res$src == receptor_name, ]
        n_hop <- 1
        while (nrow(ggi_res1) > 0 & n_hop <= max_n_hop) {
            target_name <- sample(x = 1:nrow(ggi_res1), size = 1)
            ggi_res1 <- ggi_res1[target_name, ]
            if (ggi_res1$dest %in% names(tf_score)) {
                tf_score[ggi_res1$dest] <- tf_score[ggi_res1$dest] + 1
            }
            ggi_res1 <- ggi_res[ggi_res$src == ggi_res1$dest, ]
            n_hop <- n_hop + 1
        }
    }
    tf_score <- as.numeric(tf_score/10000)
}

.get_tf_path <- function(ggi_res, tf_gene, tf_hop, receptor) {
    tf_path <- NULL
    ggi_res1 <- ggi_res[ggi_res$dest == tf_gene & ggi_res$hop == tf_hop, ]
    if (tf_hop > 1) {
        tf_hop_new <- tf_hop - 1
        for (i in tf_hop_new:1) {
            ggi_res2 <- ggi_res[ggi_res$dest %in% ggi_res1$src & ggi_res$hop == i, ]
            ggi_res1 <- ggi_res1[ggi_res1$src %in% ggi_res2$dest, ]
            ggi_res2 <- ggi_res2[ggi_res2$dest %in% ggi_res1$src, ]
            if (i == tf_hop_new) {
                tf_path <- rbind(tf_path, ggi_res1, ggi_res2)
            } else {
                tf_path <- rbind(tf_path, ggi_res2)
            }
            ggi_res1 <- ggi_res2
        }
    } else {
        tf_path <- ggi_res1
    }
    tf_path_new <- NULL
    ggi_res1 <- tf_path[tf_path$src == receptor & tf_path$hop == 1, ]
    if (tf_hop > 1) {
        for (i in 2:tf_hop) {
            ggi_res2 <- tf_path[tf_path$src %in% ggi_res1$dest & tf_path$hop == i, ]
            ggi_res2 <- ggi_res2[ggi_res2$src %in% ggi_res1$dest, ]
            if (i == 2) {
                tf_path_new <- rbind(tf_path_new, ggi_res1, ggi_res2)
            } else {
                tf_path_new <- rbind(tf_path_new, ggi_res2)
            }
            ggi_res1 <- ggi_res2
        }
    } else {
        tf_path_new <- ggi_res1
    }
    ggi_res1 <- ggi_res[ggi_res$src == tf_gene & ggi_res$hop == (tf_hop + 1), ]
    tf_path_new <- rbind(tf_path_new, ggi_res1)
    tf_path_new$tf <- tf_gene
    return(tf_path_new)
}
.get_score <- function(lrdb, receptor_tf) {
    lrdb$score <- 0
    for (j in 1:nrow(lrdb)) {
        receptor_name <- lrdb$receptor[j]
        score_lr <- 1 - lrdb$lr_co_ratio_pvalue[j]
        if (receptor_name %in% receptor_tf$receptor) {
            receptor_tf_temp <- receptor_tf[receptor_tf$receptor == receptor_name, ]
            receptor_tf_temp$score_rt <- receptor_tf_temp$n_target * receptor_tf_temp$score/receptor_tf_temp$n_hop
            score_rt <- sum(receptor_tf_temp$score_rt) * (-1)
            score_rt <- 1/(1 + exp(score_rt))
            lrdb$score[j] <- sqrt(score_lr * score_rt)
        }
    }
    lrdb <- lrdb[lrdb$score > 0, ]
    if (nrow(lrdb) == 0) {
        return("NA")
    } else {
        return(lrdb)
    }
}
createDeepTalk <- function(st_data, st_meta, species, if_st_is_sc, spot_max_cell, celltype = NULL) {
    if (is(st_data, "data.frame")) {
        st_data <- methods::as(as.matrix(st_data), "dgCMatrix")
    }
    if (is(st_data, "matrix")) {
        st_data <- methods::as(st_data, "dgCMatrix")
    }
    if (!is(st_data, "dgCMatrix")) {
        stop("st_data must be a data.frame or matrix or dgCMatrix!")
    }
    if (!is.data.frame(st_meta)) {
        stop("st_meta is not a data frame!")
    }
    # check st_data and st_meta
    if (if_st_is_sc) {
        if (!all(c("cell", "x", "y") == colnames(st_meta))) {
            stop("Please provide a correct st_meta data.frame!")
        }
        if (!all(colnames(st_data) == st_meta$cell)) {
            stop("colnames(st_data) is not consistent with st_meta$cell!")
        }
        st_type <- "single-cell"
        spot_max_cell <- 1
    } else {
        if (!all(c("spot", "x", "y") == colnames(st_meta))) {
            stop("Please provide a correct st_meta data.frame!")
        }
        if (!all(colnames(st_data) == st_meta$spot)) {
            stop("colnames(st_data) is not consistent with st_meta$spot!")
        }
        st_type <- "spot"
    }
    if (is.null(spot_max_cell)) {
        stop("Please provide the spot_max_cell!")
    }
    st_data <- as.matrix(st_data)
    st_data <- st_data[which(rowSums(st_data) > 0), ]
    if (nrow(st_data) == 0) {
        stop("No expressed genes in st_data!")
    }
    # st_meta
    st_meta$nFeatures <- as.numeric(apply(st_data, 2, .percent_cell))
    st_meta$label <- "-"
    st_meta$cell_num <- spot_max_cell
    st_meta$celltype <- "unsure"
    if (!is.null(celltype)) {
        if (!is.character(celltype)) {
            stop("celltype must be a character with length equal to ST data!")
        }
        if (length(celltype) != nrow(st_meta)) {
            stop("Length of celltype must be equal to nrow(st_meta)!")
        }
        celltype_new <- .rename_chr(celltype)
        warning_info <- .show_warning(celltype, celltype_new)
        if (!is.null(warning_info)) {
            warning(warning_info)
        }
        st_meta$celltype <- celltype_new
        if_skip_dec_celltype <- TRUE
    } else {
        if_skip_dec_celltype <- FALSE
    }
    st_meta[, 1] <- .rename_chr(st_meta[, 1])
    colnames(st_data) <- st_meta[,1]
    # generate DeepTalk object
    object <- new("DeepTalk", data = list(rawdata = st_data), meta = list(rawmeta = st_meta),
        para = list(species = species, st_type = st_type, spot_max_cell = spot_max_cell, if_skip_dec_celltype = if_skip_dec_celltype))
    return(object)
}

find_lr_path <- function(object, lrpairs, pathways, max_hop = NULL, if_doParallel = T, use_n_cores = NULL) {
    # check input data
    #cat(crayon::cyan("Checking input data", "\n"))

    if (!all(c("ligand", "receptor", "species") %in% names(lrpairs))) {
        stop("Please provide a correct lrpairs data.frame! See demo_lrpairs()!")
    }
    if (!all(c("src", "dest", "pathway", "source", "type", "src_tf", "dest_tf", "species") %in%
        names(pathways))) {
        stop("Please provide a correct pathways data.frame! See demo_pathways()!")
    }
    if (is.null(use_n_cores)) {
        n_cores <- parallel::detectCores()
        n_cores <- floor(n_cores/4)
        if (n_cores < 2) {
            if_doParallel <- FALSE
        }
    } else {
        n_cores <- use_n_cores
    }
    if (if_doParallel) {
        cl <- parallel::makeCluster(n_cores)
        doParallel::registerDoParallel(cl)
    }
    st_data <- .get_st_data(object)
    species <- object@para$species
    lrpair <- lrpairs[lrpairs$species == species, ]
    lrpair <- lrpair[lrpair$ligand %in% rownames(st_data) & lrpair$receptor %in% rownames(st_data), ]
    if (nrow(lrpair) == 0) {
        stop("No ligand-recepotor pairs found in st_data!")
    }
    pathways <- pathways[pathways$species == species, ]
    pathways <- pathways[pathways$src %in% rownames(st_data) & pathways$dest %in% rownames(st_data), ]
    ggi_tf <- pathways[, c("src", "dest", "src_tf", "dest_tf")]
    ggi_tf <- unique(ggi_tf)
    lrpair <- lrpair[lrpair$receptor %in% ggi_tf$src, ]
    if (nrow(lrpair) == 0) {
        stop("No downstream target genes found for receptors!")
    }
    #cat(crayon::cyan("Begin to filter lrpairs and pathways", "\n"))
    if (is.null(max_hop)) {
        if (species == "Mouse") {
            max_hop <- 4
        } else {
            max_hop <- 3
        }
    }
    ### find receptor-tf
    res_ggi <- NULL
    receptor_name <- unique(lrpair$receptor)
    if (if_doParallel) {
        res_ggi <- foreach::foreach(i=1:length(receptor_name), .combine = "c", .packages = "Matrix") %dopar% {
            ggi_res <- NULL
            lr_receptor <- receptor_name[i]
            ggi_tf1 <- ggi_tf[ggi_tf$src == lr_receptor, ]
            ggi_tf1 <- unique(ggi_tf1[ggi_tf1$dest %in% rownames(st_data), ])
            if (nrow(ggi_tf1) > 0) {
                n <- 0
                ggi_tf1$hop <- n + 1
                while (n <= max_hop) {
                    ggi_res <- rbind(ggi_res, ggi_tf1)
                    ggi_tf1 <- ggi_tf[ggi_tf$src %in% ggi_tf1$dest, ]
                    ggi_tf1 <- unique(ggi_tf1[ggi_tf1$dest %in% rownames(st_data), ])
                    if (nrow(ggi_tf1) == 0) {
                        break
                    }
                    ggi_tf1$hop <- n + 2
                    n <- n + 1
                }
                ggi_res <- unique(ggi_res)
                ggi_res_yes <- ggi_res[ggi_res$src_tf == "YES" | ggi_res$dest_tf == "YES", ]
                if (nrow(ggi_res_yes) > 0) {
                    lr_receptor
                }
            }
        }
    } else {
        for (i in 1:length(receptor_name)) {
            ggi_res <- NULL
            lr_receptor <- receptor_name[i]
            ggi_tf1 <- ggi_tf[ggi_tf$src == lr_receptor, ]
            ggi_tf1 <- unique(ggi_tf1[ggi_tf1$dest %in% rownames(st_data), ])
            if (nrow(ggi_tf1) > 0) {
                n <- 0
                ggi_tf1$hop <- n + 1
                while (n <= max_hop) {
                    ggi_res <- rbind(ggi_res, ggi_tf1)
                    ggi_tf1 <- ggi_tf[ggi_tf$src %in% ggi_tf1$dest, ]
                    ggi_tf1 <- unique(ggi_tf1[ggi_tf1$dest %in% rownames(st_data), ])
                    if (nrow(ggi_tf1) == 0) {
                        break
                    }
                    ggi_tf1$hop <- n + 2
                    n <- n + 1
                }
                ggi_res <- unique(ggi_res)
                ggi_res_yes <- ggi_res[ggi_res$src_tf == "YES" | ggi_res$dest_tf == "YES", ]
                if (nrow(ggi_res_yes) > 0) {
                    res_ggi <- c(res_ggi, lr_receptor)
                }
            }
        }
    }
    if (length(res_ggi) == 0) {
        stop("No downstream transcriptional factors found for receptors!")
    }
    cat(crayon::green("***Done***", "\n"))
    if (if_doParallel) {
        doParallel::stopImplicitCluster()
        parallel::stopCluster(cl)
    }
    lrpair <- lrpair[lrpair$receptor %in% res_ggi, ]
    if (nrow(lrpair) == 0) {
        stop("No ligand-recepotor pairs found!")
    }
    object@lr_path <- list(lrpairs = lrpair, pathways = pathways)
    object@para$max_hop <- max_hop
    if_skip_dec_celltype <- object@para$if_skip_dec_celltype
    if (if_skip_dec_celltype) {
        st_meta <- object@meta$rawmeta
        st_dist <- .st_dist(st_meta)
        object@dist <- st_dist
    }
    return(object)
}


CellCellCOM <- function(filepath1,filepath2,filepath3,filepath4,filepath5,filepath6,filepath7) {


    load(filepath1)
    load(filepath2)

    lrpairs <- read.csv(arg2)
    my_dataframe <- read.csv(filepath3)
    colnames(my_dataframe)[1] <- "cell"
    my_column <- my_dataframe[, 1] + 1
    my_column_with_prefix <- paste("C", my_column, sep = "")
    my_dataframe[, 1] <- my_column_with_prefix
    st_coef <- my_dataframe[, -c(1:5)]
    cellname <- colnames(st_coef)
    for (i in 1:nrow(st_coef)) {
        st_coef1 <- as.numeric(st_coef[i, ])
        if (max(st_coef1) > 0.5) {
            my_dataframe$celltype[i] <- cellname[which(st_coef1 == max(st_coef1))]
        }
    }
    my_dataframenew<-my_dataframe[,c('cell','x','y','celltype')]
    write.csv(my_dataframenew, file = filepath4)

    meta_data <- read.csv(filepath4, row.names = 1)
    st_data <- read.csv(filepath5, row.names = 1)

    obj <- createDeepTalk(st_data = as.matrix(st_data),
                         st_meta = meta_data[, -4],
                         species = "Mouse",
                         if_st_is_sc = T,
                         spot_max_cell = 1,
                         celltype = meta_data$celltype)


    obj <- find_lr_path(object = obj, lrpairs = lrpairs, pathways = pathways)


    obj <- dec_cci_all1(object = obj,use_n_cores = 20)
    #save(obj, file = "cci.Rdata")
    st_meta <- obj@meta$rawmeta
    st_data <- obj@data$rawdata
    cellname <- unique(st_meta$celltype)
    cellname <- cellname[order(cellname)]
    lrpair<-obj@lrpair
    cell_pair_all <- data.frame(cell_sender = character(), cell_receiver = character(), label = character())
    #cat(crayon::cyan("Begin to write LR pairs", "\n"))
    for (j in 1:nrow(lrpair)) {
        celltype_sender = lrpair[j,]$celltype_sender
        celltype_receiver = lrpair[j,]$celltype_receiver
        ligand = lrpair[j,]$ligand
        receptor = lrpair[j,]$receptor

        cell_pair <- obj@cellpair
        cell_pair <- cell_pair[[paste0(celltype_sender, " -- ", celltype_receiver)]]
        st_data <- st_data[, st_meta$cell]
        st_meta$ligand <- as.numeric(st_data[ligand, ])
        st_meta$receptor <- as.numeric(st_data[receptor, ])
        st_meta_ligand <- st_meta[st_meta$celltype == celltype_sender & st_meta$ligand > 0, ]
        st_meta_receptor <- st_meta[st_meta$celltype == celltype_receiver & st_meta$receptor > 0, ]

        cell_pair1 <- cell_pair[cell_pair$cell_sender %in% st_meta_ligand$cell, ]
        cell_pair1 <- cell_pair[cell_pair$cell_receiver %in% st_meta_receptor$cell, ]
        cell_pair$celltype_sender<-celltype_sender
        cell_pair$celltype_receiver<-celltype_receiver
        cell_pair$ligand<-ligand
        cell_pair$receptor<-receptor

        cell_pair$id<-rownames(cell_pair)
        cell_pair1$id<-rownames(cell_pair1)
        cell_pair1$label<-1
        cell_pair$label<-0
        cell_pair[cell_pair$id %in% cell_pair1$id,]$label<-1
        cell_pair$id <- NULL
        cell_pair_all <- rbind(cell_pair_all, cell_pair)
    }

    write.csv(cell_pair_all, file = filepath6, row.names = FALSE)
    write.csv(obj@lr_path$lrpairs, file = filepath7, row.names = FALSE)

    return(cell_pair_all)
}


args <- commandArgs(trailingOnly = TRUE)

arg1 <- args[1]
arg2 <- args[2]

filepath1 <- file.path(arg1, "/pathways.rda")
filepath2 <- file.path(arg1, "/geneinfo.rda")
filepath3 <- file.path(arg1, "/st_obs.csv")
filepath4 <- file.path(arg1, "/st_meta.csv")
filepath5 <- file.path(arg1, "/ad_st_new.csv")
filepath6 <- file.path(arg1, "/cell_pair_all.csv")
filepath7 <- file.path(arg1, "/lrpairs_pre.csv")
cell_pair_all <- CellCellCOM(filepath1,filepath2,filepath3,filepath4,filepath5,filepath6,filepath7)