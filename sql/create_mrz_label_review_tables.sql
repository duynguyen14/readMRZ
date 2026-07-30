IF OBJECT_ID(N'dbo.readmrz_label_items', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.readmrz_label_items (
        id BIGINT IDENTITY(1,1) NOT NULL PRIMARY KEY,
        source_key NVARCHAR(700) NOT NULL,
        source_file_name NVARCHAR(512) NULL,
        status VARCHAR(32) NOT NULL CONSTRAINT DF_readmrz_label_items_status DEFAULT ('labeled'),
        review_status VARCHAR(32) NOT NULL CONSTRAINT DF_readmrz_label_items_review_status DEFAULT ('pending'),
        split VARCHAR(20) NULL,

        image_file_name NVARCHAR(512) NULL,
        label_file_name NVARCHAR(512) NULL,
        rejected_image_file_name NVARCHAR(512) NULL,
        rejected_label_file_name NVARCHAR(512) NULL,

        bbox_x1 FLOAT NULL,
        bbox_y1 FLOAT NULL,
        bbox_x2 FLOAT NULL,
        bbox_y2 FLOAT NULL,
        yolo_label NVARCHAR(256) NULL,
        mrz_lines_json NVARCHAR(MAX) NULL,
        mrz_score FLOAT NULL,

        ocr_ms INT NULL,
        elapsed_ms INT NULL,
        fingerprint_size BIGINT NULL,
        fingerprint_mtime_ns BIGINT NULL,
        ocr_engine NVARCHAR(128) NULL,
        ocr_config_json NVARCHAR(MAX) NULL,
        error_message NVARCHAR(2000) NULL,

        processed_at DATETIME2 NULL,
        imported_at DATETIME2 NOT NULL CONSTRAINT DF_readmrz_label_items_imported_at DEFAULT SYSUTCDATETIME(),
        reviewed_at DATETIME2 NULL,
        updated_at DATETIME2 NOT NULL CONSTRAINT DF_readmrz_label_items_updated_at DEFAULT SYSUTCDATETIME(),

        CONSTRAINT UQ_readmrz_label_items_source_key UNIQUE (source_key),
        CONSTRAINT CK_readmrz_label_items_status CHECK (status IN ('labeled', 'no_mrz', 'error')),
        CONSTRAINT CK_readmrz_label_items_review_status CHECK (review_status IN ('pending', 'approved', 'rejected'))
    );
END;

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_readmrz_label_items_review_queue'
      AND object_id = OBJECT_ID(N'dbo.readmrz_label_items')
)
BEGIN
    CREATE INDEX IX_readmrz_label_items_review_queue
    ON dbo.readmrz_label_items (status, review_status, id)
    INCLUDE (source_key, split, image_file_name, label_file_name, mrz_score, ocr_ms);
END;

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_readmrz_label_items_split'
      AND object_id = OBJECT_ID(N'dbo.readmrz_label_items')
)
BEGIN
    CREATE INDEX IX_readmrz_label_items_split
    ON dbo.readmrz_label_items (split, review_status);
END;

IF OBJECT_ID(N'dbo.readmrz_label_review_history', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.readmrz_label_review_history (
        id BIGINT IDENTITY(1,1) NOT NULL PRIMARY KEY,
        label_item_id BIGINT NOT NULL,
        source_key NVARCHAR(700) NOT NULL,
        decision VARCHAR(32) NOT NULL,
        note NVARCHAR(1000) NULL,
        reviewed_at DATETIME2 NOT NULL CONSTRAINT DF_readmrz_label_review_history_reviewed_at DEFAULT SYSUTCDATETIME(),
        CONSTRAINT FK_readmrz_label_review_history_item
            FOREIGN KEY (label_item_id) REFERENCES dbo.readmrz_label_items(id),
        CONSTRAINT CK_readmrz_label_review_history_decision CHECK (decision IN ('approved', 'rejected'))
    );
END;

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_readmrz_label_review_history_source_key'
      AND object_id = OBJECT_ID(N'dbo.readmrz_label_review_history')
)
BEGIN
    CREATE INDEX IX_readmrz_label_review_history_source_key
    ON dbo.readmrz_label_review_history (source_key, reviewed_at DESC);
END;
