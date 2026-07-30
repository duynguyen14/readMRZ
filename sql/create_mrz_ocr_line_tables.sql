IF OBJECT_ID(N'dbo.readmrz_label_items', N'U') IS NULL
BEGIN
    THROW 50001, 'Missing dbo.readmrz_label_items. Run tools/create_mrz_label_review_tables.py first.', 1;
END;

IF COL_LENGTH(N'dbo.readmrz_label_items', N'mrz_line_extract_status') IS NULL
BEGIN
    ALTER TABLE dbo.readmrz_label_items
    ADD mrz_line_extract_status VARCHAR(32) NULL,
        mrz_line_extract_count INT NULL,
        mrz_line_extract_error NVARCHAR(2000) NULL,
        mrz_line_extracted_at DATETIME2 NULL;
END;

IF OBJECT_ID(N'dbo.readmrz_ocr_line_items', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.readmrz_ocr_line_items (
        id BIGINT IDENTITY(1,1) NOT NULL PRIMARY KEY,
        label_item_id BIGINT NOT NULL,
        source_key NVARCHAR(700) NOT NULL,
        split VARCHAR(20) NULL,
        line_index INT NOT NULL,

        line_image_file_name NVARCHAR(700) NOT NULL,
        line_image_width INT NULL,
        line_image_height INT NULL,

        line_bbox_x1 FLOAT NULL,
        line_bbox_y1 FLOAT NULL,
        line_bbox_x2 FLOAT NULL,
        line_bbox_y2 FLOAT NULL,
        doc_orientation_angle INT NOT NULL CONSTRAINT DF_readmrz_ocr_line_items_doc_angle DEFAULT (0),

        ocr_text NVARCHAR(160) NULL,
        normalized_text NVARCHAR(160) NULL,
        final_text NVARCHAR(160) NULL,
        ocr_score FLOAT NULL,
        mrz_likeness FLOAT NULL,

        review_status VARCHAR(32) NOT NULL CONSTRAINT DF_readmrz_ocr_line_items_review_status DEFAULT ('pending'),
        error_message NVARCHAR(2000) NULL,

        created_at DATETIME2 NOT NULL CONSTRAINT DF_readmrz_ocr_line_items_created_at DEFAULT SYSUTCDATETIME(),
        reviewed_at DATETIME2 NULL,
        updated_at DATETIME2 NOT NULL CONSTRAINT DF_readmrz_ocr_line_items_updated_at DEFAULT SYSUTCDATETIME(),

        CONSTRAINT FK_readmrz_ocr_line_items_label_item
            FOREIGN KEY (label_item_id) REFERENCES dbo.readmrz_label_items(id),
        CONSTRAINT UQ_readmrz_ocr_line_items_label_line UNIQUE (label_item_id, line_index),
        CONSTRAINT CK_readmrz_ocr_line_items_review_status CHECK (review_status IN ('pending', 'approved', 'rejected'))
    );
END;

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_readmrz_ocr_line_items_review_queue'
      AND object_id = OBJECT_ID(N'dbo.readmrz_ocr_line_items')
)
BEGIN
    CREATE INDEX IX_readmrz_ocr_line_items_review_queue
    ON dbo.readmrz_ocr_line_items (review_status, id)
    INCLUDE (source_key, split, line_index, line_image_file_name);
END;

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_readmrz_ocr_line_items_source_key'
      AND object_id = OBJECT_ID(N'dbo.readmrz_ocr_line_items')
)
BEGIN
    CREATE INDEX IX_readmrz_ocr_line_items_source_key
    ON dbo.readmrz_ocr_line_items (source_key, line_index);
END;

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_readmrz_label_items_line_extract'
      AND object_id = OBJECT_ID(N'dbo.readmrz_label_items')
)
BEGIN
    CREATE INDEX IX_readmrz_label_items_line_extract
    ON dbo.readmrz_label_items (review_status, mrz_line_extract_status, id)
    INCLUDE (source_key, split, image_file_name);
END;
