library(foreach)
library(Matrix)
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

.generate_newmeta_cell <- function(newmeta, st_ndata, sc_ndata, sc_celltype, iter_num, n_cores, if_doParallel) {
    newmeta_spotname <- unique(newmeta$spot)
    newmeta_cell <- NULL
    cat(crayon::cyan("Generating single-cell data for each spot", "\n"))
    if (if_doParallel) {
        cl <- parallel::makeCluster(n_cores)
        doParallel::registerDoParallel(cl)
        newmeta_cell <- foreach::foreach (i = 1:length(newmeta_spotname), .combine = "rbind", .packages = "Matrix", .export = ".generate_newmeta_spot") %dopar% {
            spot_name <- newmeta_spotname[i]
            .generate_newmeta_spot(spot_name, newmeta, st_ndata, sc_ndata, sc_celltype, iter_num)
        }
        doParallel::stopImplicitCluster()
        parallel::stopCluster(cl)
    } else {
        for (i in 1:length(newmeta_spotname)) {
            spot_name <- newmeta_spotname[i]
            newmeta_spot <- .generate_newmeta_spot(spot_name, newmeta, st_ndata, sc_ndata, sc_celltype, iter_num)
            newmeta_cell <- rbind(newmeta_cell, newmeta_spot)
        }
    }
    newmeta_cell$cell <- paste0("C", 1:nrow(newmeta))
    newmeta_cell <- newmeta_cell[, c(8, 4, 5, 3:1, 7, 6)]
    return(newmeta_cell)
}

.generate_newmeta_spot <- function(spot_name, newmeta, st_ndata, sc_ndata, sc_celltype, iter_num) {
    newmeta_spot <- newmeta[newmeta$spot == spot_name, ]
    spot_ndata <- as.numeric(st_ndata[, spot_name])
    # random sampling
    score_cor <- NULL
    spot_cell_id <- list()
    for (k in 1:iter_num) {
        spot_cell_id_k <- NULL
        for (j in 1:nrow(newmeta_spot)) {
            spot_celltype <- newmeta_spot$celltype[j]
            sc_celltype1 <- sc_celltype[sc_celltype$celltype == spot_celltype, "cell"]
            sc_celltype1 <- sc_celltype1[sample(x = 1:length(sc_celltype1), size = 1)]
            spot_cell_id_k <- c(spot_cell_id_k, sc_celltype1)
        }
        if (length(spot_cell_id_k) == 1) {
            spot_ndata_pred <- as.numeric(sc_ndata[, spot_cell_id_k])
        } else {
            spot_ndata_pred <- as.numeric(rowSums(sc_ndata[, spot_cell_id_k]))
        }
        spot_ndata_cor <- cor(spot_ndata, spot_ndata_pred)
        score_cor <- c(score_cor, spot_ndata_cor)
        spot_cell_id[[k]] <- spot_cell_id_k
    }
    spot_cell_id <- spot_cell_id[[which.max(score_cor)]]
    newmeta_spot$cell_id <- spot_cell_id
    newmeta_spot$cor <- max(score_cor)
    return(newmeta_spot)
}
sc_data <- read.csv("./ad_sc.csv")
sc_data <- data.frame(row.names = sc_data$X, sc_data[, -1])
st_data <- read.csv("./ad_ge.csv")
st_data <- data.frame(row.names = st_data$X, st_data[, -1])
colnames(st_data) <- gsub("\\.", "-", colnames(st_data))
colnames(st_data) <- toupper(colnames(st_data))
newmeta <- read.csv("./st_obs.csv")
colnames(newmeta)[1] <- "cell"
newmeta$cell <- newmeta$cell + 1
newmeta$cell <- paste0("C", newmeta$cell)
#colnames(newmeta)[4] <- "spot"
colnames(newmeta)[5] <- "celltype"
newmeta$spot <-newmeta$centroids 
newmeta$spot <- sub("_.*", "", newmeta$spot)
newmeta$celltype <- gsub(" ", "_", newmeta$celltype)
scmeta <- read.csv("./sc_obs.csv")

sc_celltype <- scmeta$cell_subclass
sc_celltype_new <- .rename_chr(sc_celltype)
sc_celltype <- data.frame(cell = colnames(sc_data), celltype = sc_celltype_new, stringsAsFactors = F)

newmeta_cell <- .generate_newmeta_cell(newmeta, st_data, sc_data, sc_celltype, 100, n_cores=10, if_doParallel=T)

newdata <- sc_data[, newmeta_cell$cell_id]
colnames(newdata) <- newmeta_cell$cell
write.csv(newdata, "newdata.csv", row.names = TRUE)
